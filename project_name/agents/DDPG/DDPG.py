import sys
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax.training.train_state import TrainState
from project_name.utils import MemoryState
from project_name.agents import AgentBase
from project_name.agents.DDPG import get_DDPG_config, ContinuousQNetwork, ScannedRNN, DeterministicPolicy
import flashbax as fbx
import optax
from functools import partial
from typing import Any, NamedTuple
import flax
from flax.core.frozen_dict import freeze
import rlax
from project_name.utils import TransitionFlashbax, TrainStateExt


class TrainStateDDPG(NamedTuple):
    critic_state: TrainStateExt
    actor_state: TrainStateExt
    n_updates: int = 0


class DDPGAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_DDPG_config()
        self.env = env
        self.env_params = env_params
        self.utils = utils
        self.critic_network = ContinuousQNetwork(config=config)  # TODO separate RNN and normal
        self.actor_network = DeterministicPolicy(env.action_space().shape, config=config,
                                                 action_scale=self.agent_config.ACTION_SCALE)

        key, _key = jrandom.split(key)

        init_x = (jnp.zeros((1, config.NUM_ENVS, *utils.observation_space(env, env_params))),
                 (jnp.zeros((1, config.NUM_ENVS, env.action_space().shape))))

        init_actor_x = (jnp.zeros((1, config.NUM_ENVS, *utils.observation_space(env, env_params))),
                      jnp.zeros((1, config.NUM_ENVS)),
                      )

        self.critic_network_params = self.critic_network.init(_key, init_x[0], init_x[1])
        self.actor_network_params = self.actor_network.init(_key, init_actor_x)

        self.agent_config.NUM_MINIBATCHES = min(self.config.NUM_ENVS, self.agent_config.NUM_MINIBATCHES)

        self.buffer = fbx.make_flat_buffer(max_length=self.agent_config.BUFFER_SIZE,
                                                           min_length=self.agent_config.BATCH_SIZE,
                                                           sample_batch_size=self.agent_config.BATCH_SIZE,
                                                           add_sequences=True,
                                                           add_batch_size=None)  # TODO should this be the real batch size?

        self.eps_scheduler = optax.linear_schedule(init_value=self.agent_config.EPS_START,
                                                   end_value=self.agent_config.EPS_FINISH,
                                                   transition_steps=self.agent_config.EPS_DECAY * config.NUM_EPISODES,
                                                   )

        def linear_schedule(count):  # TODO put this somewhere better
            frac = (1.0 - (count // (
                    self.agent_config.NUM_MINIBATCHES * self.agent_config.UPDATE_EPOCHS)) / config.NUM_EPISODES)
            return self.agent_config.LR * frac

    def create_train_state(self):
        return (TrainStateDDPG(critic_state=TrainStateExt.create(apply_fn=self.critic_network.apply,
                                                                      params=self.critic_network_params,
                                                                      target_params=self.critic_network_params,
                                                                      tx=optax.adam(self.agent_config.LR_CRITIC, eps=1e-5)),
                              actor_state=TrainStateExt.create(apply_fn=self.actor_network.apply,
                                                               params=self.actor_network_params,
                                                               target_params=self.actor_network_params,
                                                               tx=optax.adam(self.agent_config.LR_ACTOR, eps=1e-5))),
                self.buffer.init(
                    TransitionFlashbax(done=jnp.zeros((self.config.NUM_ENVS,), dtype=bool),
                                  action=jnp.zeros((self.config.NUM_ENVS,
                                                    self.env.action_space().shape), dtype=jnp.float64),
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
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        _, action = train_state.actor_state.apply_fn(train_state.actor_state.params, ac_in)  # TODO no rnn for now
        key, _key = jrandom.split(key)
        action += jnp.clip(jrandom.normal(_key, action.shape) * self.agent_config.ACTION_SCALE * self.agent_config.EXPLORATION_NOISE,
                           -self.env.params.A_MAX, self.env.params.A_MAX)

        # action = rlax.add_ornstein_uhlenbeck_noise(_key, action, )

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

        def critic_loss(critic_target_params, critic_params, actor_target_params, batch):
            obs = batch.experience.first.obs
            action = batch.experience.first.action
            reward = batch.experience.first.reward
            done = batch.experience.first.done
            nobs = batch.experience.second.obs
            ndone = batch.experience.second.done

            _, action_pred = train_state.actor_state.apply_fn(actor_target_params, (nobs, ndone))
            action_pred = jnp.clip(action_pred, -self.env.params.A_MAX, self.env.params.A_MAX)
            target_val = train_state.critic_state.apply_fn(critic_target_params, nobs, action_pred)

            y_expected = reward + (1 - done) * self.agent_config.GAMMA * jnp.squeeze(target_val, axis=-1)  # TODO do I need stop gradient?

            y_pred = train_state.critic_state.apply_fn(critic_params, obs, action)

            loss_critic = optax.losses.huber_loss(jnp.squeeze(y_pred, axis=-1), y_expected) / 1.0  # same as smooth l1 loss ?

            return jnp.mean(loss_critic)

        critic_loss, grads = jax.value_and_grad(critic_loss, argnums=1, has_aux=False)(train_state.critic_state.target_params,
                                                                                      train_state.critic_state.params,
                                                                                      train_state.actor_state.target_params,
                                                                                      batch
                                                                                      )

        new_critic_state = train_state.critic_state.apply_gradients(grads=grads)
        # train_state = train_state._replace(critic_state=train_state.critic_state.apply_gradients(grads=grads))

        def policy_loss(critic_params, actor_params, batch):
            obs = batch.experience.first.obs
            done = batch.experience.first.done

            _, action_pred = train_state.actor_state.apply_fn(actor_params, (obs, done))

            q_val = train_state.critic_state.apply_fn(critic_params, obs, action_pred)

            loss_actor = -jnp.mean(q_val)

            def critic_mean(critic_params, obs, dpg_a_t):
                logits = train_state.critic_state.apply_fn(critic_params, obs, dpg_a_t)

                return jnp.mean(logits)

            dpg_a_t = action_pred
            dq_da = jax.vmap(jax.vmap(jax.grad(critic_mean, argnums=2, has_aux=False), in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(critic_params, obs, dpg_a_t)
            dqda_clipping = None  # can also be 1
            rlax_loss = jax.vmap(jax.vmap(rlax.dpg_loss, in_axes=(0, 0, None)), in_axes=(0, 0, None))(dpg_a_t, dq_da, dqda_clipping)
            rlax_loss = jnp.mean(rlax_loss)

            return rlax_loss  # loss_actor

        actor_loss, grads = jax.value_and_grad(policy_loss, argnums=1, has_aux=False)(new_critic_state.params,
                                                                                    train_state.actor_state.params,
                                                                                    batch
                                                                                    )

        new_actor_state = train_state.actor_state.apply_gradients(grads=grads)
        # train_state = train_state._replace(actor_state=train_state.actor_state.apply_gradients(grads=grads))

        # update target network
        new_critic_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
                                        lambda new_critic_state:
                                        new_critic_state.replace(target_params=optax.incremental_update(new_critic_state.params,
                                                                    new_critic_state.target_params,
                                                                    self.agent_config.TAU)),
                                        lambda new_critic_state: new_critic_state, operand=new_critic_state)
        new_actor_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
                                        lambda new_actor_state:
                                        new_actor_state.replace(target_params=optax.incremental_update(new_actor_state.params,
                                                                    new_actor_state.target_params,
                                                                    self.agent_config.TAU)),
                                        lambda new_actor_state: new_actor_state, operand=new_actor_state)
        # TODO above needs reworking if possible

        train_state = train_state._replace(critic_state=new_critic_state,
                                           actor_state=new_actor_state,
                                           n_updates=train_state.n_updates+1)

        info = {"value_loss": jnp.mean(critic_loss),
                "policy_loss": jnp.mean(actor_loss)
                }

        return train_state, mem_state, env_state, info, key