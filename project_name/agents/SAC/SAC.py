import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax
import chex

from functools import partial
from typing import Any, Tuple, NamedTuple
import distrax
import rlax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState
from project_name.agents import AgentBase
from project_name.agents.SAC import get_SAC_config, DiscreteActor, DiscreteDoubleSoftQNetwork, ContinuousActor, ContinuousDoubleSoftQNetwork
import optax
import flashbax as fbx
from project_name.vapor_stuff.utils import TransitionNoInfo
import sys
from project_name.utils import TrainStateExt, TransitionFlashbax


class TrainStateSAC(NamedTuple):
    critic_state: TrainStateExt
    actor_state: TrainStateExt
    log_alpha: flax.core.FrozenDict
    alpha_opt_state: optax.OptState
    n_updates: int = 0


class SACAgent(AgentBase):
    def __init__(self, env, env_params, key, config, utils):
        self.config = config
        self.agent_config = get_SAC_config()
        self.env = env
        self.env_params = env_params
        self.utils = utils

        key, actor_key, critic_key = jrandom.split(key, 3)

        if self.config.DISCRETE:
            self.critic_network = DiscreteDoubleSoftQNetwork(env.action_space(env_params).shape)
            self.actor_network = DiscreteActor(env.action_space(env_params).n)
            self.critic_params = self.critic_network.init(critic_key,
                                                          jnp.zeros((1, config.NUM_ENVS, *utils.observation_space(env, env_params))))
        else:
            self.critic_network = ContinuousDoubleSoftQNetwork()
            self.actor_network = ContinuousActor(env.action_space(env_params).shape, self.agent_config)
            self.critic_params = self.critic_network.init(critic_key,
                                                          jnp.zeros((1, config.NUM_ENVS,
                                                                     *utils.observation_space(env, env_params))),
                                                          jnp.zeros((1, config.NUM_ENVS,
                                                                     env.action_space(env_params).shape))
                                                          )

        self.actor_params = self.actor_network.init(actor_key,
                                                    jnp.zeros((1, config.NUM_ENVS,
                                                               *utils.observation_space(env, env_params))))

        self.buffer = fbx.make_flat_buffer(max_length=self.agent_config.BUFFER_SIZE,
                                                           min_length=self.agent_config.BATCH_SIZE,
                                                           sample_batch_size=self.agent_config.BATCH_SIZE,
                                                           add_sequences=True,
                                                           add_batch_size=None)

        self.ACTION_SCALE = (env.action_space(env_params).high - env.action_space(env_params).low) / 2.0
        self.ACTION_BIAS =(env.action_space(env_params).high + env.action_space(env_params).low) / 2.0

        self.target_entropy = -env.action_space(env_params).shape

        self.alpha_optimiser = optax.adam(self.agent_config.ALPHA_LR)

    def create_train_state(self):
        log_alpha=jnp.asarray(jnp.log(self.agent_config.INIT_ALPHA), dtype=jnp.float32)
        return (TrainStateSAC(critic_state=TrainStateExt.create(apply_fn=self.critic_network.apply,
                                               params=self.critic_params,
                                               target_params=self.critic_params,
                                               tx=optax.chain(
                                                   optax.inject_hyperparams(optax.adam)(self.agent_config.LR, eps=1e-4)),
                                               ),
                              actor_state=TrainState.create(apply_fn=self.actor_network.apply,
                                        params=self.actor_params,
                                        tx=optax.chain(optax.inject_hyperparams(optax.adam)(self.agent_config.LR, eps=1e-4)),
                                        ),
                              log_alpha=log_alpha,
                              alpha_opt_state=self.alpha_optimiser.init(log_alpha)),
                self.buffer.init(
                    TransitionFlashbax(done=jnp.zeros((self.config.NUM_ENVS,), dtype=bool),
                                       action=jnp.zeros((self.config.NUM_ENVS,
                                                         self.env.action_space().shape), dtype=jnp.float32),
                                       reward=jnp.zeros((self.config.NUM_ENVS,)),
                                       obs=jnp.zeros((self.config.NUM_ENVS,
                                                      *self.utils.observation_space(self.env, self.env_params)),
                                                     dtype=jnp.float32),
                                       # TODO is it always an int for the obs?
                                       )))

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: chex.PRNGKey):
        pi = train_state.actor_state.apply_fn(train_state.actor_state.params, ac_in[0])
        key, _key = jrandom.split(key)
        x_t = pi.sample(seed=_key)
        y_t = jnp.tanh(x_t)

        action = y_t * self.ACTION_SCALE + self.ACTION_BIAS

        return mem_state, action, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch_LNZ, unused_2):
        train_state, mem_state, env_state, ac_in, key = runner_state

        mem_state = self.buffer.add(mem_state, TransitionFlashbax(done=traj_batch_LNZ.done,
                                                              action=traj_batch_LNZ.action,
                                                              reward=traj_batch_LNZ.reward,
                                                              obs=traj_batch_LNZ.obs,
                                                              ))

        key, _key = jrandom.split(key)
        batch = self.buffer.sample(mem_state, _key)

        log_alpha = train_state.log_alpha
        alpha = jnp.exp(log_alpha)

        def critic_loss(critic_target_params, critic_params, actor_params, batch, key):
            obs_BNO = batch.experience.first.obs
            action_BNA = batch.experience.first.action
            reward_BN = batch.experience.first.reward
            done_BN = batch.experience.first.done
            nobs_BNO = batch.experience.second.obs

            pi = train_state.actor_state.apply_fn(actor_params, nobs_BNO)
            key, _key = jrandom.split(key)
            x_t_BNA = pi.sample(seed=_key)
            y_t_BNA = jnp.tanh(x_t_BNA)
            naction_BNA = y_t_BNA * self.ACTION_SCALE + self.ACTION_BIAS

            nlog_prob_BNA = pi.log_prob(x_t_BNA)
            nlog_prob_BNA = nlog_prob_BNA - jnp.log(self.agent_config.ACTION_SCALE * (1 - y_t_BNA ** 2) + 1e-6)
            nlog_prob_BN1 = jnp.sum(nlog_prob_BNA, axis=-1, keepdims=True)

            qf_next_target_BN12 = train_state.critic_state.apply_fn(critic_target_params, nobs_BNO, naction_BNA)
            qf_next_target_BN1 = jnp.min(qf_next_target_BN12, axis=-1) - alpha  * nlog_prob_BN1

            next_q_value_BN = reward_BN + (1 - done_BN) * self.agent_config.GAMMA * jnp.squeeze(qf_next_target_BN1, axis=-1)

            # use Q-values only for the taken actions
            qf_values_BN12 = train_state.critic_state.apply_fn(critic_params, obs_BNO, action_BNA)

            def mse_loss(preds, targets):
                return 0.5 * jnp.mean(jnp.square(preds - targets))

            qf_loss = jax.vmap(mse_loss, in_axes=(2, 2))(jnp.squeeze(qf_values_BN12, axis=2),
                                                         jnp.repeat(next_q_value_BN[..., None], 2, axis=-1))

            return jnp.sum(qf_loss)

        critic_loss, grads = jax.value_and_grad(critic_loss, argnums=1, has_aux=False)(
            train_state.critic_state.target_params,
            train_state.critic_state.params,
            train_state.actor_state.params,
            batch,
            key
        )

        new_critic_state = train_state.critic_state.apply_gradients(grads=grads)

        def actor_loss(critic_params, actor_params, batch, key):
            obs_BNO = batch.experience.first.obs

            pi = train_state.actor_state.apply_fn(actor_params, obs_BNO)
            key, _key = jrandom.split(key)
            x_t_BNA = pi.sample(seed=_key)
            y_t_BNA = jnp.tanh(x_t_BNA)
            action_BNA = y_t_BNA * self.ACTION_SCALE + self.ACTION_BIAS

            log_prob_BNA = pi.log_prob(x_t_BNA)
            log_prob_BNA = log_prob_BNA - jnp.log(self.agent_config.ACTION_SCALE * (1 - y_t_BNA ** 2) + 1e-6)
            log_prob_BN1 = jnp.sum(log_prob_BNA, axis=-1, keepdims=True)

            min_qf_values_BN12 = train_state.critic_state.apply_fn(critic_params, obs_BNO, action_BNA)
            min_qf_values_BN1 = jnp.min(min_qf_values_BN12, axis=-1)

            return jnp.mean(((alpha * log_prob_BN1) - min_qf_values_BN1))

        actor_loss, grads = jax.value_and_grad(actor_loss, argnums=1)(train_state.critic_state.params,
                                                           train_state.actor_state.params,
                                                           batch,
                                                           key
                                                           )
        new_actor_state = train_state.actor_state.apply_gradients(grads=grads)

        def alpha_loss(log_alpha, actor_params, batch, key):
            obs_BNO = batch.experience.first.obs

            pi = train_state.actor_state.apply_fn(actor_params, obs_BNO)
            key, _key = jrandom.split(key)
            x_t_BNA = pi.sample(seed=_key)
            y_t_BNA = jnp.tanh(x_t_BNA)

            log_prob_BNA = pi.log_prob(x_t_BNA)
            log_prob_BNA = log_prob_BNA - jnp.log(self.agent_config.ACTION_SCALE * (1 - y_t_BNA ** 2) + 1e-6)
            log_prob_BN1 = jnp.sum(log_prob_BNA, axis=-1, keepdims=True)

            alpha_loss = -jnp.exp(log_alpha) * (log_prob_BN1 + self.target_entropy)

            return jnp.mean(alpha_loss)

        alpha_loss, grads = jax.value_and_grad(alpha_loss, argnums=0)(train_state.log_alpha,
                                                                      train_state.actor_state.params,
                                                                      batch,
                                                                      key)

        alpha_updates, new_alpha_opt_state = self.alpha_optimiser.update(grads, train_state.alpha_opt_state)
        new_log_alpha = optax.apply_updates(train_state.log_alpha, alpha_updates)
        alpha = jnp.exp(new_log_alpha)

        # update target network
        new_critic_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
                                        lambda new_critic_state:
                                        new_critic_state.replace(
                                            target_params=optax.incremental_update(new_critic_state.params,
                                                                                   new_critic_state.target_params,
                                                                                   self.agent_config.TAU)),
                                        lambda new_critic_state: new_critic_state, operand=new_critic_state)

        train_state = train_state._replace(critic_state=new_critic_state,
                                           actor_state=new_actor_state,
                                           log_alpha=new_log_alpha,
                                           alpha_opt_state=new_alpha_opt_state,
                                           n_updates=train_state.n_updates+1)

        info = {"value_loss": jnp.mean(critic_loss),
                "policy_loss": jnp.mean(actor_loss),
                "alpha": alpha,
                "alpha_loss": jnp.mean(alpha_loss),
                }

        return train_state, mem_state, env_state, info, key