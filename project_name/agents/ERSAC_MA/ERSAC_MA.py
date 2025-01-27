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
from project_name.agents.ERSAC_MA import get_ERSAC_MA_config, ActorCritic, EnsembleNetwork
import numpy as np
import distrax
import flax
import rlax


class TrainStateERSAC(NamedTuple):
    ac_state: TrainState
    ens_state: Any  # TODO how to update this?
    log_tau: Any
    tau_opt_state: Any


class TrainStateRP(TrainState):  # TODO check gradients do not update the static prior
    static_prior_params: flax.core.FrozenDict


class ERSAC_MAAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_ERSAC_MA_config()
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

        self.log_tau = jnp.asarray(jnp.log(self.agent_config.INIT_TAU), dtype=jnp.float32)
        # self.log_tau = jnp.asarray(self.agent_config.INIT_TAU, dtype=jnp.float32)
        self.tau_optimiser = optax.adam(learning_rate=self.agent_config.TAU_LR)

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
        return (TrainStateERSAC(ac_state=TrainState.create(apply_fn=self.network.apply,
                                                           params=self.network_params,
                                                           tx=self.tx),
                                ens_state=jax.vmap(create_ensemble_state, in_axes=(0))(ensemble_keys),
                                log_tau=self.log_tau,
                                tau_opt_state=self.tau_optimiser.init(self.log_tau)),
                MemoryState(hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
                            extras={
                                "values": jnp.zeros((self.config.NUM_ENVS, 1)),
                                "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
                            })
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        mem_state = mem_state._replace(extras={
            "values": jnp.zeros((self.config.NUM_ENVS, 1)),
            "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
        },
            hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
        )
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: TrainStateERSAC, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        pi, value, action_logits = train_state.ac_state.apply_fn(train_state.ac_state.params, ac_in[0])
        # TODO can I resort done to be better stored
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        # action = jnp.ones_like(action)  # TODO for testing randomized actions

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward_noise(self, ens_state: TrainStateRP, obs: chex.Array, actions: chex.Array, key) -> chex.Array:
        ensemble_obs = jnp.broadcast_to(obs, (self.agent_config.NUM_ENSEMBLE, *obs.shape))
        ensemble_action = jnp.broadcast_to(actions, (self.agent_config.NUM_ENSEMBLE, *actions.shape))

        def single_reward_noise(ens_state: TrainStateRP, obs: chex.Array, action: chex.Array) -> chex.Array:
            rew_pred = ens_state.apply_fn({"params": {"_net": ens_state.params,
                                                      "_prior_net": ens_state.static_prior_params}},
                                          obs, jnp.expand_dims(action, axis=-1))
            return rew_pred

        ensembled_reward = jax.vmap(single_reward_noise)(ens_state,
                                                         ensemble_obs,
                                                         ensemble_action)

        ensembled_reward = jnp.var(ensembled_reward, axis=0)

        return ensembled_reward

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch):
        traj_batch = jax.tree_map(lambda x: x[:, agent], traj_batch)
        train_state, mem_state, env_state, ac_in, key = runner_state

        # key, _key = jrandom.split(key)
        # mask = jrandom.binomial(_key, 1, self._mask_prob, self._num_ensemble)  # TODO version of jax has no binomial?
        mask = np.random.binomial(1, self.agent_config.MASK_PROB, self.agent_config.NUM_ENSEMBLE)
        # TODO add noise generation

        obs = jnp.concatenate((traj_batch.obs, jnp.zeros((1, *traj_batch.obs.shape[1:]))), axis=0)

        state_action_reward_noise = self._get_reward_noise(train_state.ens_state, traj_batch.obs, traj_batch.action, key)

        def ac_loss(params, trajectory, obs, tau_params, state_action_reward_noise):
            tau = jnp.exp(tau_params)

            _, values, logits = train_state.ac_state.apply_fn(params, obs)
            policy_dist = distrax.Categorical(logits=logits[:-1])  # ensure this is the same as the network distro
            log_prob = policy_dist.log_prob(trajectory.action)

            td_lambda = jax.vmap(rlax.td_lambda, in_axes=(1, 1, 1, 1, None), out_axes=1)
            k_estimate = td_lambda(values[:-1],
                                   trajectory.reward + (jnp.squeeze(state_action_reward_noise, axis=-1) / (2 * tau)),
                                   (1 - trajectory.done) * self.agent_config.GAMMA,
                                   values[1:],
                                   self.agent_config.TD_LAMBDA,
                                   )

            value_loss = jnp.mean(jnp.square(values[:-1] - jax.lax.stop_gradient(k_estimate - tau * log_prob)))

            entropy = policy_dist.entropy()

            policy_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(k_estimate - values[:-1] - tau * entropy))

            return policy_loss + value_loss, entropy

        (pv_loss, entropy), grads = jax.value_and_grad(ac_loss, has_aux=True, argnums=0)(train_state.ac_state.params,
                                                                                         traj_batch,
                                                                                         obs,
                                                                                         train_state.log_tau,
                                                                                         state_action_reward_noise)
        train_state = train_state._replace(ac_state=train_state.ac_state.apply_gradients(grads=grads))

        def tau_loss(tau_params, entropy, state_action_reward_noise):
            tau = jnp.exp(tau_params)

            tau_loss = jnp.squeeze(state_action_reward_noise, axis=-1) / (2 * tau) + (tau * entropy)

            return jnp.mean(tau_loss)

        tau_loss_val, tau_grads = jax.value_and_grad(tau_loss, has_aux=False, argnums=0)(train_state.log_tau,
                                                                                         entropy,
                                                                                         state_action_reward_noise)
        tau_updates, new_tau_opt_state = self.tau_optimiser.update(tau_grads, train_state.tau_opt_state)
        new_tau_params = optax.apply_updates(train_state.log_tau, tau_updates)
        train_state = train_state._replace(log_tau=new_tau_params, tau_opt_state=new_tau_opt_state)

        def train_ensemble(ens_state: TrainStateRP, obs, actions, rewards, mask):
            def reward_predictor_loss(rp_params, prior_params, obs, actions, rewards, mask):
                rew_pred = ens_state.apply_fn({"params": {"_net": rp_params,
                                                      "_prior_net": prior_params}},
                                              obs, jnp.expand_dims(actions, axis=-1))
                # rew_pred += reward_noise_scale * jnp.expand_dims(z_t, axis=-1)
                return 0.5 * jnp.mean(mask * jnp.square(jnp.squeeze(rew_pred, axis=-1) - rewards)), rew_pred
                # return jnp.mean(jnp.zeros((2))), rew_pred

            (ensemble_loss, rew_pred), grads = jax.value_and_grad(reward_predictor_loss, argnums=0, has_aux=True)(ens_state.params,
                                                                                                                  ens_state.static_prior_params,
                                                                                        obs,
                                                                                        actions,
                                                                                        rewards,
                                                                                        mask)
            ens_state = ens_state.apply_gradients(grads=grads)

            return ensemble_loss, ens_state, rew_pred

        ensemble_obs = jnp.broadcast_to(traj_batch.obs, (self.agent_config.NUM_ENSEMBLE, *traj_batch.obs.shape))
        ensemble_action = jnp.broadcast_to(traj_batch.action,
                                           (self.agent_config.NUM_ENSEMBLE, *traj_batch.action.shape))
        ensemble_reward = jnp.broadcast_to(traj_batch.reward,
                                           (self.agent_config.NUM_ENSEMBLE, *traj_batch.reward.shape))
        ensemble_mask = np.random.binomial(1, self.agent_config.MASK_PROB, (ensemble_reward.shape))

        ensembled_loss, ens_state, rew_pred = jax.vmap(train_ensemble)(train_state.ens_state,
                                                             ensemble_obs,
                                                             ensemble_action,
                                                             ensemble_reward,
                                                             ensemble_mask)
        train_state = train_state._replace(ens_state=ens_state)

        info = {"ac_loss": pv_loss,
                "tau_loss": tau_loss_val,
                "tau": jnp.exp(new_tau_params),
                }
        for ensemble_id in range(self.agent_config.NUM_ENSEMBLE):
            info[f"Ensemble_{ensemble_id}_Reward_Pred_pv"] = rew_pred[ensemble_id, 6, 6]  # index random step and random batch
            info[f"Ensemble_{ensemble_id}_Loss"] = ensembled_loss[ensemble_id]

        return train_state, mem_state, env_state, info, key
