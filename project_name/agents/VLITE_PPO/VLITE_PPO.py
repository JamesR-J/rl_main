import sys
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple
import jax.random as jrandom
from functools import partial
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState
from project_name.agents import AgentBase
import chex
from project_name.agents.VLITE_PPO import get_VLITE_PPO_config, ActorCritic, EnsembleNetwork, binomial
import numpy as np
import distrax
import flax
import rlax
from distrax._src.utils import math


class TrainStateVLITE(NamedTuple):
    ac_state: TrainState
    ens_state: TrainState


class TrainStateRP(TrainState):
    static_prior_params: flax.core.FrozenDict


class VLITE_PPOAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_VLITE_PPO_config()
        self.env = env
        self.env_params = env_params
        self.network = ActorCritic(env.action_space().n, config=config, agent_config=self.agent_config)
        self.rp_network = EnsembleNetwork(config=config, agent_config=self.agent_config)

        if self.config.CNN:
            self._init_x = jnp.zeros((1, config.NUM_ENVS, *env.observation_space(env_params).shape))
        else:
            self._init_x = jnp.zeros((1, config.NUM_ENVS, utils.observation_space(env, env_params)))

        key, _key = jrandom.split(key)
        self.network_params = self.network.init(_key, self._init_x)

        self.key = key

        self.tx = optax.adam(self.agent_config.LR)

    def create_train_state(self):
        def create_ensemble_state(key: chex.PRNGKey) -> TrainState:  # TODO is this the best place to put it all?
            key, _key = jrandom.split(key)
            rp_params = self.rp_network.init(_key, self._init_x,
                                             jnp.zeros((1, self.config.NUM_ENVS, 1)))["params"],
            reward_state = TrainStateRP.create(apply_fn=self.rp_network.apply,
                                             params=rp_params[0]["_net"],  # TODO unsure why it needs a 0 index here?
                                             static_prior_params=rp_params[0]["_prior_net"],
                                             tx=optax.adam(self.agent_config.ENS_LR))
            return reward_state

        ensemble_keys = jrandom.split(self.key, self.agent_config.NUM_ENSEMBLE)
        return (TrainStateVLITE(ac_state=TrainState.create(apply_fn=self.network.apply,
                                                           params=self.network_params,
                                                           tx=self.tx),
                                ens_state=jax.vmap(create_ensemble_state, in_axes=(0))(ensemble_keys)),
                MemoryState(hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
                            extras={
                                "action_logits": jnp.zeros((self.config.NUM_ENVS, 1, self.env.action_space().n)),
                                "values": jnp.zeros((self.config.NUM_ENVS, 1)),
                                "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
                            })
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        mem_state = mem_state._replace(extras={
            "action_logits": jnp.zeros((self.config.NUM_ENVS, 1, self.env.action_space().n)),
            "values": jnp.zeros((self.config.NUM_ENVS, 1)),
            "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
        },
            hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
        )
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: TrainStateVLITE, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        pi, value, action_logits = train_state.ac_state.apply_fn(train_state.ac_state.params, ac_in[0])
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        mem_state.extras["action_logits"] = jnp.swapaxes(action_logits, 0, 1)
        mem_state = mem_state._replace(extras=mem_state.extras)

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward_noise(self, ens_state: TrainStateRP, obs: chex.Array, actions: chex.Array) -> chex.Array:
        def single_reward_noise(ens_state: TrainStateRP, obs: chex.Array, action: chex.Array) -> chex.Array:
            rew_pred = ens_state.apply_fn({"params": {"_net": ens_state.params,
                                                      "_prior_net": ens_state.static_prior_params}},
                                          obs, jnp.expand_dims(action, axis=-1))
            return rew_pred

        ensembled_reward = jax.vmap(single_reward_noise, in_axes=(0, None, None))(ens_state,
                                                         obs,
                                                         actions)

        ensembled_reward = self.agent_config.UNCERTAINTY_SCALE * jnp.std(ensembled_reward, axis=0)
        ensembled_reward = jnp.minimum(ensembled_reward, 1.0)

        return ensembled_reward

    def _reward_noise_over_actions(self, ens_state: TrainStateRP, obs: chex.Array) -> chex.Array:  # TODO sort this oot
        actions = jnp.expand_dims(jnp.arange(0, self.env.action_space().n, step=1), axis=(-1, -2))
        actions = jnp.broadcast_to(actions, (actions.shape[0], obs.shape[0], obs.shape[1]))

        reward_over_actions = jax.vmap(self._get_reward_noise, in_axes=(None, None, 0))(ens_state, obs, actions)
        reward_over_actions = jnp.swapaxes(jnp.squeeze(reward_over_actions, axis=-1), 0, 1)
        reward_over_actions = jnp.swapaxes(reward_over_actions, 1, 2)

        return reward_over_actions

    @partial(jax.jit, static_argnums=(0,))
    def _entropy_loss_fn(self, logits_t, uncertainty_t):
        log_pi = jax.nn.log_softmax(logits_t)
        pi_times_log_pi = math.mul_exp(log_pi, log_pi)

        return -jnp.sum(pi_times_log_pi * uncertainty_t, axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch, unused_2):
        traj_batch = jax.tree_map(lambda x: x[:, agent], traj_batch)
        # print(traj_batch)
        train_state, mem_state, env_state, ac_in, key = runner_state
        _, last_val, _ = train_state.ac_state.apply_fn(train_state.ac_state.params, ac_in[0])  # TODO 1 is dones
        last_val = last_val.squeeze(axis=0)

        state_action_reward_noise = self._get_reward_noise(train_state.ens_state, traj_batch.obs, traj_batch.action)
        state_reward_noise = self._reward_noise_over_actions(train_state.ens_state, traj_batch.obs)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + self.agent_config.GAMMA * next_value * (1 - done) - value
                gae = (delta + self.agent_config.GAMMA * self.agent_config.GAE_LAMBDA * (1 - done) * gae)
                return (gae, value), gae

            _, advantages = jax.lax.scan(_get_advantages,
                                         (jnp.zeros_like(last_val), last_val),
                                         traj_batch,
                                         reverse=True,
                                         unroll=16,
                                         )
            return advantages, advantages + traj_batch.value

        traj_batch = traj_batch._replace(reward=traj_batch.reward+jnp.squeeze(state_action_reward_noise, axis=-1))  # TODO check this
        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets, state_reward_noise = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value, _ = train_state.ac_state.apply_fn(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-self.agent_config.CLIP_EPS,
                                                                                            self.agent_config.CLIP_EPS)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(
                        where=(1 - traj_batch.done))

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (jnp.clip(ratio,
                                            1.0 - self.agent_config.CLIP_EPS,
                                            1.0 + self.agent_config.CLIP_EPS,
                                            ) * gae)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean(where=(1 - traj_batch.done))
                    # entropy = pi.entropy().mean(where=(1 - traj_batch.done))
                    entropy = jax.vmap(self._entropy_loss_fn, in_axes=1, out_axes=1)(pi.logits, state_reward_noise).mean(where=(1 - traj_batch.done))

                    total_loss = (loss_actor
                                  + self.agent_config.VF_COEF * value_loss
                                  - self.agent_config.ENT_COEF * entropy
                                  )

                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.ac_state.params, traj_batch, advantages, targets)
                train_state = train_state._replace(ac_state=train_state.ac_state.apply_gradients(grads=grads))

                return train_state, total_loss

            train_state, traj_batch, advantages, targets, key = update_state
            key, _key = jrandom.split(key)

            permutation = jrandom.permutation(_key, self.config.NUM_ENVS)
            batch = (traj_batch,
                     advantages,
                     targets,
                     state_reward_noise)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(
                jnp.reshape(x, [x.shape[0], self.agent_config.NUM_MINIBATCHES, -1] + list(x.shape[2:]), ), 1, 0, ),
                                                 shuffled_batch, )

            train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)

            update_state = (train_state,
                            traj_batch,
                            advantages,
                            targets,
                            key,
                            )
            return update_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, key)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, self.agent_config.UPDATE_EPOCHS)
        train_state, traj_batch, advantages, targets, key = update_state

        def train_ensemble(ens_state: TrainStateRP, obs, actions, rewards, mask):
            def reward_predictor_loss(rp_params, prior_params, obs, actions, rewards, mask):
                rew_pred = ens_state.apply_fn({"params": {"_net": rp_params,
                                                          "_prior_net": prior_params}},
                                              obs, jnp.expand_dims(actions, axis=-1))
                # rew_pred += reward_noise_scale * jnp.expand_dims(z_t, axis=-1)
                return 0.5 * jnp.mean(mask * jnp.square(jnp.squeeze(rew_pred, axis=-1) - rewards)), rew_pred
                # return jnp.mean(jnp.zeros((2))), rew_pred

            (ensemble_loss, rew_pred), grads = jax.value_and_grad(reward_predictor_loss, argnums=0, has_aux=True)(
                ens_state.params,
                ens_state.static_prior_params,
                obs,
                actions,
                rewards,
                mask)
            ens_state = ens_state.apply_gradients(grads=grads)

            return ensemble_loss, ens_state, rew_pred

        # ensemble_mask = np.random.binomial(1, self.agent_config.MASK_PROB, (self.agent_config.NUM_ENSEMBLE,
        #                                                                     *traj_batch.action.shape))
        key, _key = jrandom.split(key)
        ensemble_mask = binomial(_key, 1, self.agent_config.MASK_PROB, (self.agent_config.NUM_ENSEMBLE,
                                                                        *traj_batch.action.shape))

        ensembled_loss, ens_state, rew_pred = jax.vmap(train_ensemble, in_axes=(0, None, None, None, 0))(
            train_state.ens_state,
            traj_batch.obs,
            traj_batch.action,
            traj_batch.reward,
            ensemble_mask)
        train_state = train_state._replace(ens_state=ens_state)

        info = {"value_loss": jnp.mean(loss_info[1][0]),
                "actor_loss": jnp.mean(loss_info[1][1]),
                "entropy": jnp.mean(loss_info[1][2]),
                }
        for ensemble_id in range(self.agent_config.NUM_ENSEMBLE):
            info[f"Ensemble_{ensemble_id}_Reward_Pred_pv"] = rew_pred[ensemble_id, 6, 6]  # index random step and random batch
            info[f"Ensemble_{ensemble_id}_Loss"] = ensembled_loss[ensemble_id]

        return train_state, mem_state, env_state, info, key
