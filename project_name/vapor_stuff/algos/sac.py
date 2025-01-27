import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax
import chex

from functools import partial
from typing import Any, Tuple
import distrax

from project_name.vapor_stuff.algos.network_deepsea import SoftQNetwork, Actor, DoubleSoftQNetwork
from flax.training.train_state import TrainState
import optax
import flashbax as fbx
from project_name.vapor_stuff.utils import TransitionNoInfo
import sys


class TrainStateCritic(TrainState):  # TODO check gradients do not update target_params
    target_params: flax.core.FrozenDict


class SAC:
    def __init__(self, env, env_params, key, config):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.actor_network = Actor(action_dim=env.action_space(env_params).n)
        self.critic_network = DoubleSoftQNetwork(action_dim=env.action_space(env_params).n)

        key, actor_key, critic_key = jrandom.split(key, 3)

        self.actor_params = self.actor_network.init(actor_key,
                                                    jnp.zeros((1, *env.observation_space(env_params).shape, 1)))
        self.critic_params = self.critic_network.init(critic_key,
                                                      jnp.zeros((1, *env.observation_space(env_params).shape, 1)))

        self.per_buffer = fbx.make_prioritised_flat_buffer(max_length=config.BUFFER_SIZE,
                                                           min_length=config.BATCH_SIZE,
                                                           sample_batch_size=config.BATCH_SIZE,
                                                           add_sequences=True,
                                                           add_batch_size=None,
                                                           priority_exponent=config.REPLAY_PRIORITY_EXP,
                                                           device=config.DEVICE)

    def create_train_state(self, key: chex.Array) -> Tuple[
        type(flax.training.train_state), TrainStateCritic, Any, Any, chex.PRNGKey]:  # TODO imrpove checks any
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

        buffer_state = self.per_buffer.init(
            TransitionNoInfo(state=jnp.zeros((*self.env.observation_space(self.env_params).shape, 1)),
                             action=jnp.zeros((1), dtype=jnp.int32),
                             reward=jnp.zeros((1)),
                             ensemble_reward=jnp.zeros((1)),
                             done=jnp.zeros((1), dtype=bool),
                             ))

        return actor_state, critic_state, critic_state, buffer_state, key  # dummy critic state extra

    @partial(jax.jit, static_argnums=(0,))
    def act(self, actor_params: dict, obs: chex.Array, key: chex.PRNGKey) -> Tuple[
        chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
        key, _key = jrandom.split(key)
        logits = self.actor_network.apply(actor_params, obs)
        policy_dist = distrax.Categorical(logits=logits)
        action = policy_dist.sample(seed=_key)
        # log_prob = policy_dist.log_prob(action)
        action_probs = policy_dist.probs

        z = action_probs == 0.0
        z = z * 1e-8
        log_prob = jnp.log(action_probs + z)

        return action, log_prob, action_probs, key

    @partial(jax.jit, static_argnums=(0,))
    def update_target_network(self, critic_state: TrainStateCritic) -> TrainStateCritic:
        critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params,
                                                                                   critic_state.target_params,
                                                                                   self.config.TAU)
                                            )

        return critic_state

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state):
        actor_state, critic_state, _, buffer_state, _, _, _, key = runner_state
        key, _key = jrandom.split(key)
        batch = self.per_buffer.sample(buffer_state, _key)

        # CRITIC training
        def critic_loss(actor_params, critic_params, critic_target_params, batch, key):
            obs = batch.experience.first.state
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.state

            _, next_state_log_pi, next_state_action_probs, key = self.act(actor_params, nobs, key)

            qf_next_target = self.critic_network.apply(critic_target_params, nobs)
            qf_next_target = jnp.min(qf_next_target, axis=-1)

            qf_next_target = next_state_action_probs * (qf_next_target)  #   - (self.config.ALPHA * next_state_log_pi))
            # TODO above have removed entropy from policy evaluation

            # adapt Q-target for discrete Q-function
            qf_next_target = jnp.sum(qf_next_target, axis=1, keepdims=True)

            # VAPOR-LITE
            next_q_value = reward + (1 - done) * self.config.GAMMA * (qf_next_target)
            # TODO should use done or discount? I think these match up

            # use Q-values only for the taken actions
            qf_values = self.critic_network.apply(critic_params, obs)
            qf_values = jnp.min(qf_values, axis=-1)
            qf1_a_values = jnp.take_along_axis(qf_values, action, axis=1)

            td_error = qf1_a_values - next_q_value  # TODO ensure this okay as other stuff vmaps over time?

            # mse loss below
            # qf_loss = 0.5 * jnp.mean(jnp.square(td_error))  # TODO check this is okay?

            # Get the importance weights.
            importance_weights = (1. / batch.priorities).astype(jnp.float32)
            importance_weights **= self.config.IMPORTANCE_SAMPLING_EXP  # TODO what is this val?
            importance_weights /= jnp.max(importance_weights)

            # reweight
            qf_loss = 0.5 * jnp.mean(importance_weights * jnp.square(td_error))
            new_priorities = jnp.abs(td_error) + 1e-7

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

            min_qf_values = self.critic_network.apply(critic_params, obs)  # TODO ensure it uses the right params
            min_qf_values = jnp.min(min_qf_values, axis=-1)

            _, log_pi, action_probs, _ = self.act(actor_params, obs, key)

            return jnp.mean(action_probs * ((self.config.ALPHA * log_pi) - min_qf_values))

        actor_loss, grads = jax.value_and_grad(actor_loss, argnums=0)(actor_state.params,
                                                           critic_state.params,
                                                           batch,
                                                           key
                                                           )
        actor_state = actor_state.apply_gradients(grads=grads)

        # TODO maybe add the alpha tuning?

        return actor_state, critic_state, critic_state, buffer_state, actor_loss, critic_loss, jnp.zeros(1), key
