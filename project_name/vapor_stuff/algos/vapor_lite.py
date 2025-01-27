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

from project_name.vapor_stuff.algos.network_deepsea import SoftQNetwork, Actor, RandomisedPrior, DoubleSoftQNetwork
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
                                                      jnp.zeros((1, *env.observation_space(env_params).shape, 1)))

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
                                        tx=optax.adam(self.config.LR),
                                        )
        critic_state = TrainStateCritic.create(apply_fn=self.critic_network.apply,  # TODO check this actually works
                                               params=self.critic_params,
                                               target_params=self.critic_params,
                                               # TODO does this need copying? worth checking to ensure params and target arent the same
                                               tx=optax.adam(self.config.LR),
                                               )

        def create_reward_state(key: chex.PRNGKey) -> TrainStateRP:  # TODO is this the best place to put it all?
            key, _key = jrandom.split(key)
            # rp_params = \
                # self.rp_network.init(_key,
                #                      (jnp.zeros((1, *self.env.observation_space(self.env_params).shape, 1)),
                #                       jnp.zeros((1, 1))))["params"]
            rp_params = self.rp_network.init(_key,
                                 (jrandom.uniform(_key, (1, *self.env.observation_space(self.env_params).shape, 1), minval=0.0, maxval=1.0),
                                  jrandom.uniform(_key, (1, 1), minval=0.0, maxval=1.0)))["params"]
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
        # action_probs = policy_dist.prob(action)
        action_probs = policy_dist.probs
        z = action_probs == 0.0
        z = z * 1e-8
        log_prob = jnp.log(action_probs + z)  # TODO idk if this is right but eyo

        # pi_s, log_pi_s = self.actor_network.apply(actor_params, obs)
        # action = jnp.argmax(pi_s, axis=1)  # TODO is it an argmax or is it a sample?
        #
        # return action, log_pi_s, pi_s, key

        return action, log_prob, action_probs, logits, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward_noise(self, ensrpr_state, obs: chex.Array, actions: chex.Array, key) -> chex.Array:
        ensemble_obs = jnp.repeat(obs[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)
        ensemble_action = jnp.repeat(actions[jnp.newaxis, :], self.config.NUM_ENSEMBLE, axis=0)

        ensemble_keys = jrandom.split(key, self.config.NUM_ENSEMBLE)

        # bootstraps = jnp.stack(jnp.array([utils.generate_bootstrap(ensemble_keys[i], obs.shape[0]) for i in range(1,
        #                                                                                                     self.config.NUM_ENSEMBLE + 1)]))  # TODO sort out the keys here, also this is MEGA slow need to improve
        # ensemble_obs_bootstrap = jax.vmap(jnp.take, in_axes=(None, 0, None))(obs, bootstraps, 0)
        # ensemble_action_bootstrap = jnp.expand_dims(jax.vmap(jnp.take, in_axes=(None, 0))(actions, bootstraps), axis=-1)

        ensemble_action_bootstrap, ensemble_obs_bootstrap, _ \
            = utils.bootstrap_samples(key, actions, obs, m=self.config.NUM_ENSEMBLE)

        ensemble_obs = ensemble_obs_bootstrap
        ensemble_action = ensemble_action_bootstrap

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
    def _reward_noise_over_actions(self, ensrpr_state: TrainStateRP, obs: chex.Array, key) -> chex.Array:
        # run the get_reward_noise for each action choice, can probs vmap
        actions = jnp.expand_dims(jnp.arange(0, self.env.action_space(self.env_params).n, step=1), axis=-1)
        actions = jnp.expand_dims(jnp.tile(actions, obs.shape[0]), axis=-1)

        obs = jnp.repeat(obs[jnp.newaxis, :], self.env.action_space(self.env_params).n, axis=0)

        reward_over_actions = jax.vmap(self._get_reward_noise, in_axes=(None, 0, 0, None))(ensrpr_state,
                                                                                     obs,
                                                                                     actions,
                                                                                     key)
        # reward_over_actions = jnp.sum(reward_over_actions, axis=0)  # TODO removed the layer sum
        reward_over_actions = jnp.swapaxes(jnp.squeeze(reward_over_actions, axis=-1), 0, 1)

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
            done = self.config.GAMMA * (1 - batch.experience.first.done)  # TODO put the minus here
            # nobs = batch.experience.second.state

            _, log_pi, action_probs, logits_actor, key = self.act(actor_params, obs, key)
            qf_values = self.critic_network.apply(critic_params, obs)
            qf_values = jnp.min(qf_values, axis=-1)
            qf_values = action_probs * (qf_values)
            qf_values = qf_values.sum(axis=-1)[:, jnp.newaxis]
            v_t = qf_values[1:]
            v_tm1 = qf_values[:-1]

            discounts = done[1:]

            state_action_reward_noise = self._get_reward_noise(ensrpr_state, obs, action, key)
            rewards = reward[1:] + state_action_reward_noise[1:]

            rhos = utils.categorical_importance_sampling_ratios(logits_actor[:-1], logits[:-1], jnp.squeeze(action[:-1],
                                                                                                            axis=-1))

            vtrace_td_error_and_advantage = jax.vmap(utils.vtrace_td_error_and_advantage, in_axes=(1, 1, 1, 1, 1, None),
                                                     out_axes=1)
            vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, rewards, discounts, jnp.expand_dims(rhos, axis=-1), 0.9)  # TODO should have a batch dim
            # TODO think as using dones it is 1-discounts rather than just discounts, but done higher up

            pg_advs = jax.lax.stop_gradient(vtrace_returns.pg_advantage)

            td_error = vtrace_returns.errors
            # qf_loss = 0.5 * jnp.sum(jnp.square(td_error))

            # _, target_log_pi, target_action_probs, _, key = self.act(actor_params, nobs, key)
            #
            # qf_next_target = self.critic_network.apply(critic_target_params, nobs)
            # qf_next_target = jnp.min(qf_next_target, axis=-1)  # TODO what axis for this one?
            # # TODO ensure it uses the right params
            #
            # next_state_reward_noise = self._reward_noise_over_actions(ensrpr_state, nobs)
            # state_action_reward_noise = self._get_reward_noise(ensrpr_state, obs, action)
            #
            # min_qf_next_target = target_action_probs * (qf_next_target)  # - (next_state_reward_noise * target_log_pi))
            #
            # # adapt Q-target for discrete Q-function
            # min_qf_next_target = min_qf_next_target.sum(axis=-1)[:, jnp.newaxis]  # TODO what axis is this again?
            #
            # # # VAPOR-LITE
            # next_q_value = reward + state_action_reward_noise + (1 - done) * (min_qf_next_target)
            # #
            # # # use Q-values only for the taken actions
            # qf_values = self.critic_network.apply(critic_params, obs)
            # qf_values = jnp.min(qf_values, axis=-1)
            # qf_a_values = jnp.take_along_axis(qf_values, action, axis=1)
            # #
            # td_error = qf_a_values - next_q_value  # TODO ensure this okay as other stuff vmaps over time?
            #
            # _, _, _, target_logits, key = self.act(actor_params, obs, key)
            #
            # rho = utils.categorical_importance_sampling_ratios(target_logits, logits, jnp.squeeze(action, axis=-1))
            #
            # vmapped_lambda = jnp.repeat(jnp.array(0.9), qf_a_values.shape[0])
            #
            # td_error = jax.vmap(utils.vtrace)(qf_a_values,
            #                                   min_qf_next_target,
            #                                   reward + state_action_reward_noise,
            #                                   done,
            #                                   jnp.expand_dims(rho, axis=-1),
            #                                   vmapped_lambda)
            #
            # pg_advs = jnp.zeros((1))

            # mse loss below
            # qf_loss = utils.l2_loss(td_error)

            # Get the importance weights.
            importance_weights = (1. / batch.priorities).astype(jnp.float32)
            importance_weights **= self.config.IMPORTANCE_SAMPLING_EXP  # TODO what is this val?
            importance_weights /= jnp.max(importance_weights)

            # reweight
            # qf_loss = 0.5 * jnp.mean(importance_weights * jnp.square(td_error))  # TODO is this okay idk?
            # qf_loss = 0.5 * jnp.mean(jnp.square(td_error))  # TODO is this okay idk?
            qf_loss = jnp.mean(importance_weights * (0.5 * jnp.mean(jnp.square(td_error))))
            new_priorities = jnp.abs(jnp.squeeze(td_error, axis=-1)) + 1e-7

            new_priorities = jnp.concatenate((new_priorities, jnp.array([new_priorities[0]])))
            # TODO the above is super dodge loL

            return qf_loss, (new_priorities, pg_advs, logits_actor[:-1])  # to remove the last dimensions of new_priorities

        (critic_loss, (new_priorities, pg_advs, learner_logits)), grads = jax.value_and_grad(critic_loss, argnums=1, has_aux=True)(
            actor_state.params,
            critic_state.params,
            critic_state.target_params,
            batch,
            key
        )

        buffer_state = self.per_buffer.set_priorities(buffer_state, batch.indices, new_priorities)

        critic_state = critic_state.apply_gradients(grads=grads)

        def actor_loss(actor_params, critic_params, batch, pg_advs, learner_logits, key):
            actions = batch.experience.first.action
            obs = batch.experience.first.state
            dones = batch.experience.first.done

            mask = jnp.not_equal(dones[:-1], True)
            # mask = jnp.not_equal(dones[1:], True)
            mask = mask.astype(jnp.float32)  # TODO potentially?
            # TODO is it?

            tb_pg_loss_fn = jax.vmap(utils.policy_gradient_loss, in_axes=1, out_axes=0)
            pg_loss = tb_pg_loss_fn(jnp.expand_dims(learner_logits, axis=1), actions[:-1],
                                    pg_advs, mask)
            pg_loss = jnp.mean(pg_loss)

            state_reward_noise = self._reward_noise_over_actions(ensrpr_state, obs, key)

            entropy_loss = jax.vmap(utils.entropy_loss_fn, in_axes=1)(jnp.expand_dims(learner_logits, axis=1),
                                                                jnp.expand_dims(state_reward_noise[:-1], axis=1),
                                                                jnp.expand_dims(mask, axis=-1))
            ent_loss = jnp.mean(entropy_loss)

            # ent_loss_fn = jax.vmap(utils.entropy_loss, in_axes=1, out_axes=0)
            # ent_loss = ent_loss_fn(jnp.expand_dims(learner_logits, axis=1), mask)
            # ent_loss = jnp.mean(ent_loss)

            # min_qf_values = self.critic_network.apply(critic_params, obs)  # TODO ensure it uses the right params
            # min_qf_values = jnp.min(min_qf_values, axis=-1)
            #
            #
            # state_reward_noise = self._reward_noise_over_actions(ensrpr_state, obs)
            #
            # _, log_pi, action_probs, _, _ = self.act(actor_params, obs, key)
            #
            # # return -jnp.mean(action_probs * ((state_reward_noise * action_probs * log_pi) - min_qf_values))
            # return jnp.mean(action_probs * ((state_reward_noise * log_pi) - min_qf_values))
            return pg_loss + 0.01 * ent_loss, (pg_loss, ent_loss)  # TODO maybe scale it?

        (actor_loss, (pg_loss, ent_loss)), grads = jax.value_and_grad(actor_loss, argnums=0, has_aux=True)(actor_state.params,
                                                                      critic_state.params,
                                                                      batch,
                                                                      pg_advs,
                                                                      learner_logits,
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

        # ensemble_keys = jrandom.split(key, self.config.NUM_ENSEMBLE)
        #
        # bootstraps = jnp.stack(jnp.array([utils.generate_bootstrap(ensemble_keys[i], obs.shape[0]) for i in
        #                                   range(1, self.config.NUM_ENSEMBLE + 1)]))  # TOOD sort out the keys here

        ensemble_action_bootstrap, ensemble_state_bootstrap, ensemble_reward_bootstrap\
            = utils.bootstrap_samples(key, action, obs, jitter_reward, self.config.NUM_ENSEMBLE)

        # ensemble_state_bootstrap = jax.vmap(jnp.take, in_axes=(None, 0, None))(obs, bootstraps, 0)
        # ensemble_action_bootstrap = jnp.expand_dims(jax.vmap(jnp.take, in_axes=(None, 0))(action, bootstraps), axis=-1)
        # ensemble_reward_bootstrap = jnp.expand_dims(jax.vmap(jnp.take, in_axes=(None, 0))(jitter_reward, bootstraps), axis=-1)

        ensemble_state = ensemble_state_bootstrap
        ensemble_action = ensemble_action_bootstrap
        ensemble_reward = ensemble_reward_bootstrap

        ensembled_loss, ensrpr_state = jax.vmap(train_ensemble)(ensrpr_state,
                                                                ensemble_state,
                                                                ensemble_action,
                                                                ensemble_reward)

        critic_state = jax.lax.cond(critic_state.step % self.config.TARGET_UPDATE_INTERVAL == 0,
                                    lambda spec_train_state: self.update_target_network(critic_state),
                                    lambda spec_train_state: spec_train_state, operand=critic_state
                                    )

        return actor_state, critic_state, ensrpr_state, buffer_state, actor_loss, critic_loss, jnp.mean(
            ensembled_loss), pg_loss, ent_loss, key
