import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax
import chex
import sys
# import rlax

from functools import partial
from typing import Any, Tuple
import distrax

from project_name.vapor_stuff.algos.network_deepsea_lessdiscrete import SoftQNetwork, Actor, RandomisedPrior, DoubleSoftQNetwork
from flax.training.train_state import TrainState
import optax
import flashbax as fbx
from project_name.vapor_stuff.utils import TransitionNoInfo
from project_name.vapor_stuff import utils

class TrainStateCritic(TrainState):  # TODO check gradients do not update target_params
    target_params: flax.core.FrozenDict


class TrainStateRP(TrainState):  # TODO check gradients do not update the static prior
    static_prior_params: flax.core.FrozenDict


class VAPOR_Lite:
    def __init__(self, env, env_params, key, config):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.actor_network = Actor(action_dim=env.action_space(env_params).n)
        self.critic_network = DoubleSoftQNetwork(action_dim=env.action_space(env_params).n)
        self.rp_network = RandomisedPrior()

        key, actor_key, critic_key = jrandom.split(key, 3)

        self.actor_params = self.actor_network.init(actor_key,
                                                    jnp.zeros((1, *env.observation_space(env_params).shape, 1)))
        self.critic_params = self.critic_network.init(critic_key,
                                                      jnp.zeros((1, *env.observation_space(env_params).shape, 1)),
                                                      jnp.zeros((1, 1)))

        self.per_buffer = fbx.make_prioritised_flat_buffer(max_length=config.BUFFER_SIZE,
                                                           min_length=config.BATCH_SIZE,
                                                           sample_batch_size=config.BATCH_SIZE + 1,
                                                           add_sequences=True,
                                                           add_batch_size=None,
                                                           priority_exponent=config.REPLAY_PRIORITY_EXP,
                                                           device=config.DEVICE)

    def create_train_state(self, key: chex.Array) -> Tuple[
        type(flax.training.train_state), TrainStateCritic, TrainStateRP, Any, chex.PRNGKey]:  # TODO imrpove checks any
        actor_state = TrainState.create(apply_fn=self.actor_network.apply,
                                        params=self.actor_params,
                                        tx=optax.chain(optax.inject_hyperparams(optax.adam)(self.config.LR, eps=1e-4)),
                                        )
        critic_state = TrainStateCritic.create(apply_fn=self.critic_network.apply,  # TODO check this actually works
                                               params=self.critic_params,
                                               target_params=self.critic_params,
                                               # TODO does this need copying? worth checking to ensure params and target arent the same
                                               tx=optax.chain(
                                                   optax.inject_hyperparams(optax.adam)(self.config.LR, eps=1e-4)),
                                               )

        def create_reward_state(key: chex.PRNGKey) -> TrainStateRP:  # TODO is this the best place to put it all?
            key, _key = jrandom.split(key)
            rp_params = \
                self.rp_network.init(_key,
                                     (jnp.zeros((1, *self.env.observation_space(self.env_params).shape, 1)),
                                      jnp.zeros((1, 1))))["params"]
            reward_state = TrainStateRP.create(apply_fn=self.rp_network.apply,
                                               params=rp_params["trainable"],
                                               static_prior_params=rp_params["static_prior"],
                                               tx=optax.adam(self.config.LR))
            return reward_state

        ensemble_keys = jrandom.split(key, self.config.NUM_ENSEMBLE)
        ensembled_reward_state = jax.vmap(create_reward_state, in_axes=(0))(ensemble_keys)
        # TODO maybe update this to corax from yicheng

        buffer_state = self.per_buffer.init(
            TransitionNoInfo(state=jnp.zeros((*self.env.observation_space(self.env_params).shape, 1)),
                             action=jnp.zeros((1), dtype=jnp.int32),
                             reward=jnp.zeros((1)),
                             ensemble_reward=jnp.zeros((1)),
                             done=jnp.zeros((1), dtype=bool),
                             logits=jnp.zeros((self.env.action_space(self.env_params).n), dtype=jnp.float32),
                             ))

        return actor_state, critic_state, ensembled_reward_state, buffer_state, key

    @partial(jax.jit, static_argnums=(0,))
    def act(self, actor_params: dict, obs: chex.Array, key: chex.PRNGKey) -> Tuple[
        chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
        key, _key = jrandom.split(key)
        logits = self.actor_network.apply(actor_params, obs)
        policy_dist = distrax.Categorical(logits=logits)
        action = policy_dist.sample(seed=_key)
        log_prob = policy_dist.log_prob(action)
        action_probs = policy_dist.prob(action)
        # action_probs = policy_dist.probs
        # z = action_probs == 0.0
        # z = z * 1e-8
        # log_prob = jnp.log(action_probs + z)  # TODO idk if this is right but eyo

        return action, log_prob, action_probs, logits, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward_noise(self, ensrpr_state, obs: chex.Array, actions: chex.Array) -> chex.Array:
        ensemble_obs = jnp.repeat(obs[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)
        ensemble_action = jnp.repeat(actions[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)

        def single_reward_noise(ind_rpr_state: TrainStateRP, obs: chex.Array, action: chex.Array) -> chex.Array:
            rew_pred = ind_rpr_state.apply_fn({"params": {"static_prior": ind_rpr_state.static_prior_params,
                                                          "trainable": ind_rpr_state.params}},
                                              (obs, action))
            return rew_pred

        ensembled_reward = jax.vmap(single_reward_noise)(ensrpr_state,
                                                         ensemble_obs,
                                                         ensemble_action)

        ensembled_reward = self.config.SIGMA_SCALE * jnp.std(ensembled_reward, axis=0)
        ensembled_reward = jnp.minimum(ensembled_reward, 1)

        return ensembled_reward

    @partial(jax.jit, static_argnums=(0,))
    def _reward_noise_over_actions(self, ensrpr_state: TrainStateRP, obs: chex.Array) -> chex.Array:
        # run the get_reward_noise for each action choice, can probs vmap
        actions = jnp.arange(0, self.env.action_space(self.env_params).n, step=1)[:, jnp.newaxis]
        actions = jnp.tile(actions, obs.shape[0])[:, :, jnp.newaxis]

        obs = jnp.repeat(obs[jnp.newaxis, :], self.env.action_space(self.env_params).n, axis=0)

        reward_over_actions = jax.vmap(self._get_reward_noise, in_axes=(None, 0, 0))(ensrpr_state,
                                                                                     obs,
                                                                                     actions)
        # reward_over_actions = jnp.sum(reward_over_actions, axis=0)  # TODO removed the layer sum
        reward_over_actions = jnp.swapaxes(reward_over_actions[:, :, 0], 0, 1)

        return reward_over_actions

    @partial(jax.jit, static_argnums=(0,))
    def update_target_network(self, critic_state: TrainStateCritic) -> TrainStateCritic:
        critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params,
                                                                                   critic_state.target_params,
                                                                                   self.config.TAU)
                                            )

        return critic_state

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state):
        actor_state, critic_state, ensrpr_state, buffer_state, _, _, _, key = runner_state
        key, _key = jrandom.split(key)
        batch = self.per_buffer.sample(buffer_state, _key)

        # CRITIC training
        def critic_loss(actor_params, critic_params, critic_target_params, batch, key):
            obs = batch.experience.first.state
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            logits = batch.experience.first.logits
            done = self.config.GAMMA * (1 - batch.experience.first.done)
            # nobs = batch.experience.second.state

            _, log_pi, action_probs, logits_actor, key = self.act(actor_params, obs, key)
            qf_values = self.critic_network.apply(critic_params, obs, action)
            qf_values = jnp.min(qf_values, axis=-1)
            v_t = qf_values[1:]
            v_tm1 = qf_values[:-1]

            discounts = done[1:]

            state_action_reward_noise = self._get_reward_noise(ensrpr_state, obs, action)
            rewards = reward[1:] + state_action_reward_noise[1:]

            rhos = utils.categorical_importance_sampling_ratios(logits_actor[:-1], logits[:-1],
                                                                jnp.squeeze(action[:-1], axis=-1))

            vtrace_td_error_and_advantage = jax.vmap(utils.vtrace_td_error_and_advantage, in_axes=1, out_axes=1)

            vmapped_lambda = jnp.expand_dims(jnp.repeat(jnp.array(0.9), v_t.shape[0]), axis=-1)
            vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, rewards, discounts,
                                                           jnp.expand_dims(rhos, axis=-1),
                                                           vmapped_lambda)  # TODO should have a batch dim
            # TODO think as using dones it is 1-discounts rather than just discounts, done this much higher

            pg_advs = vtrace_returns.pg_advantage

            td_error = vtrace_returns.errors
            qf_loss = 0.5 * jnp.sum(jnp.square(td_error))

            # nactions, next_state_log_pi, next_state_action_probs, _, key = self.act(actor_params, nobs, key)
            # nactions = jnp.expand_dims(nactions, axis=-1)
            #
            # qf_next_target = self.critic_network.apply(critic_target_params, nobs, nactions)
            # qf_next_target = jnp.min(qf_next_target, axis=-1)
            #
            # # next_state_reward_noise = self._reward_noise_over_actions(ensrpr_state, nobs)
            # next_state_action_reward_noise = self._get_reward_noise(ensrpr_state, nobs, nactions)
            # state_action_reward_noise = self._get_reward_noise(ensrpr_state, obs, action)
            # min_qf_next_target = qf_next_target - (next_state_action_reward_noise * jnp.expand_dims(next_state_log_pi, axis=-1))
            #
            # # VAPOR-LITE
            # # next_q_value = reward + state_action_reward_noise + (1 - done) * (min_qf_next_target)
            #
            # # use Q-values only for the taken actions
            # qf_a_values = self.critic_network.apply(critic_params, obs, action)
            # qf_a_values = jnp.min(qf_a_values, axis=-1)
            #
            # # td_error = qf_a_values - next_q_value  # TODO ensure this okay as other stuff vmaps over time?
            #
            # _, _, _, target_logits, key = self.act(actor_params, obs, key)
            #
            # rho = rlax.categorical_importance_sampling_ratios(target_logits, logits, jnp.squeeze(action, axis=-1))
            #
            # td_error = jax.vmap(rlax.vtrace)(qf_a_values,
            #                                  min_qf_next_target,
            #                                  reward + state_action_reward_noise,
            #                                  done,
            #                                  jnp.expand_dims(rho, axis=-1), 0.9)
            #
            # # mse loss below
            # # qf_loss = jnp.mean(jnp.square(td_error))  # TODO check this is okay?

            # Get the importance weights.
            importance_weights = (1. / batch.priorities).astype(jnp.float32)
            importance_weights **= self.config.IMPORTANCE_SAMPLING_EXP  # TODO what is this val?
            importance_weights /= jnp.max(importance_weights)

            # reweight
            qf_loss = 0.5 * jnp.mean(importance_weights[1:] * jnp.square(jnp.squeeze(td_error, axis=-1)))
            # TODO should it be 1: or :-1
            new_priorities = jnp.abs(td_error) + 1e-7

            new_priorities = jnp.concatenate((new_priorities, jnp.array([new_priorities[0]])))
            # TODO the above is super dodge loL

            return qf_loss, new_priorities[:, 0]  # to remove the last dimensions of new_priorities

        (critic_loss, new_priorities), grads = jax.value_and_grad(critic_loss, argnums=1, has_aux=True)(
            actor_state.params,
            critic_state.params,
            critic_state.target_params,
            batch,
            key
        )

        # rb.update_priority(abs_td)
        buffer_state = self.per_buffer.set_priorities(buffer_state, batch.indices, new_priorities)

        critic_state = critic_state.apply_gradients(grads=grads)

        def actor_loss(actor_params, critic_params, batch, key):
            obs = batch.experience.first.state

            actions, log_pi, _, _, key = self.act(actor_params, obs, key)
            actions = jnp.expand_dims(actions, axis=-1)

            min_qf_values = self.critic_network.apply(critic_params, obs, actions)  # TODO ensure it uses the right params
            min_qf_values = jnp.min(min_qf_values, axis=-1)

            state_action_reward_noise = self._get_reward_noise(ensrpr_state, obs, actions)

            return jnp.mean((state_action_reward_noise * log_pi) - min_qf_values)

        actor_loss, grads = jax.value_and_grad(actor_loss, argnums=0)(actor_state.params,
                                                           critic_state.params,
                                                           batch,
                                                           key
                                                           )
        actor_state = actor_state.apply_gradients(grads=grads)

        def train_ensemble(indrpr_state, obs, actions, rewards):
            def reward_predictor_loss(rp_params, prior_params):
                rew_pred = indrpr_state.apply_fn(
                    {"params": {"static_prior": prior_params, "trainable": rp_params}}, (obs, actions))
                return 0.5 * jnp.mean(jnp.square(rew_pred - rewards))

            ensemble_loss, grads = jax.value_and_grad(reward_predictor_loss, argnums=0)(indrpr_state.params,
                                                                                        indrpr_state.static_prior_params)
            indrpr_state = indrpr_state.apply_gradients(grads=grads)

            return ensemble_loss, indrpr_state

        obs = batch.experience.first.state  # TODO bit messy so should probs clean up
        action = batch.experience.first.action
        jitter_reward = batch.experience.first.ensemble_reward

        ensemble_state = jnp.repeat(obs[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)
        ensemble_action = jnp.repeat(action[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)
        ensemble_reward = jnp.repeat(jitter_reward[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)

        ensembled_loss, ensrpr_state = jax.vmap(train_ensemble)(ensrpr_state,
                                                                     ensemble_state,
                                                                     ensemble_action,
                                                                     ensemble_reward)

        return actor_state, critic_state, ensrpr_state, buffer_state, actor_loss, critic_loss, jnp.mean(ensembled_loss), key
