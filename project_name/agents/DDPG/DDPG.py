import sys
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax.training.train_state import TrainState
from project_name.utils import MemoryState
from project_name.agents import AgentBase
from project_name.agents.DDPG import get_DDPG_config, ContinuousRNNQNetwork, ScannedRNN, DeterministicPolicy
import flashbax as fbx
import optax
from functools import partial
from typing import Any, NamedTuple
import flax


class TrainStateExt(TrainState):
    target_params: flax.core.FrozenDict


class TrainStateDDPG(NamedTuple):
    critic_state: TrainStateExt
    actor_state: TrainStateExt
    n_updates: int = 0


class TransitionDDPG(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray


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
        self.critic_network = ContinuousRNNQNetwork(config=config)  # TODO separate RNN and normal
        self.actor_network = DeterministicPolicy(env.action_space().shape, config=config,
                                                 action_scale=self.agent_config.ACTION_SCALE)

        key, _key = jrandom.split(key)
        init_hstate = ScannedRNN.initialize_carry(config.NUM_ENVS, self.agent_config.GRU_HIDDEN_DIM)

        if self.config.CNN:
            init_x = ((jnp.zeros((1, config.NUM_ENVS, utils.observation_space(env, env_params))),
                       (jnp.zeros((1, config.NUM_ENVS, env.action_space().shape)))),
                      jnp.zeros((1, config.NUM_ENVS)),
                      )
        else:
            init_x = ((jnp.zeros((1, config.NUM_ENVS, utils.observation_space(env, env_params))),
                       (jnp.zeros((1, config.NUM_ENVS, env.action_space().shape)))),
                      jnp.zeros((1, config.NUM_ENVS)),
                      )

        init_actor_x = (jnp.zeros((1, config.NUM_ENVS, utils.observation_space(env, env_params))),
                      jnp.zeros((1, config.NUM_ENVS)),
                      )

        self.critic_network_params = self.critic_network.init(_key, init_hstate, init_x)
        self.actor_network_params = self.actor_network.init(_key, init_actor_x)

        self.init_hstate = ScannedRNN.initialize_carry(config.NUM_ENVS,
                                                       self.agent_config.GRU_HIDDEN_DIM)  # TODO do we need both?

        self.agent_config.NUM_MINIBATCHES = min(self.config.NUM_ENVS, self.agent_config.NUM_MINIBATCHES)

        self.buffer = fbx.make_flat_buffer(max_length=self.agent_config.BUFFER_SIZE,
                                                           min_length=self.agent_config.BATCH_SIZE,
                                                           sample_batch_size=self.agent_config.BATCH_SIZE,
                                                           add_sequences=True,
                                                           add_batch_size=None)  # TODO should this be the real batch size?

        self.eps_scheduler = optax.linear_schedule(init_value=self.agent_config.EPS_START,
                                                   end_value=self.agent_config.EPS_FINISH,
                                                   transition_steps=self.agent_config.EPS_DECAY * config.NUM_UPDATES,
                                                   )

        def linear_schedule(count):  # TODO put this somewhere better
            frac = (1.0 - (count // (
                    self.agent_config.NUM_MINIBATCHES * self.agent_config.UPDATE_EPOCHS)) / config.NUM_UPDATES)
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
                    TransitionDDPG(done=jnp.zeros((self.config.NUM_ENVS), dtype=bool),
                                  action=jnp.zeros((self.config.NUM_ENVS,
                                                    self.env.action_space().shape), dtype=jnp.float64),
                                  reward=jnp.zeros((self.config.NUM_ENVS)),
                                  obs=jnp.zeros((self.config.NUM_ENVS,
                                                 self.utils.observation_space(self.env, self.env_params)),
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
                           -self.env.env_params.A_MAX, self.env.env_params.A_MAX)

        log_prob = jnp.zeros((*action.shape[:2],))
        value = jnp.zeros((*action.shape[:2],))  # TODO sort these out, is this same dim number as num envs, do I need this or does it just go in mem state?

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch, unused_2):
        traj_batch = jax.tree_map(lambda x: x[:, agent], traj_batch)

        train_state, mem_state, env_state, ac_in, key = runner_state

        mem_state = self.buffer.add(mem_state, TransitionDDPG(done=traj_batch.done,
                                                                 action=traj_batch.action,
                                                                 reward=traj_batch.reward,
                                                                 obs=traj_batch.obs,
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
            action_pred = jnp.clip(action_pred, -self.env.env_params.A_MAX, self.env.env_params.A_MAX)
            _, target_val = train_state.critic_state.apply_fn(critic_target_params, None, ((nobs, action_pred), ndone))

            y_expected = reward + (1 - done) * self.agent_config.GAMMA * jnp.squeeze(target_val, axis=-1)  # TODO do I need stop gradient?

            _, y_pred = train_state.critic_state.apply_fn(critic_params, None, ((obs, action), done))

            loss_critic = optax.losses.huber_loss(jnp.squeeze(y_pred, axis=-1), y_expected) / 1.0  # same as smooth l1 loss ?

            return jnp.mean(loss_critic)

        critic_loss, grads = jax.value_and_grad(critic_loss, argnums=1, has_aux=False)(train_state.critic_state.target_params,
                                                                                      train_state.critic_state.params,
                                                                                      train_state.actor_state.target_params,
                                                                                      batch
                                                                                      )

        train_state = train_state._replace(critic_state=train_state.critic_state.apply_gradients(grads=grads))

        def actor_loss(critic_params, actor_params, batch):
            obs = batch.experience.first.obs
            done = batch.experience.first.done

            _, action_pred = train_state.actor_state.apply_fn(actor_params, (obs, done))

            _, q_val = train_state.critic_state.apply_fn(critic_params, None, ((obs, action_pred), done))

            loss_actor = -jnp.mean(q_val)

            return loss_actor

        actor_loss, grads = jax.value_and_grad(actor_loss, argnums=1, has_aux=False)(train_state.critic_state.params,
                                                                                    train_state.actor_state.params,
                                                                                    batch
                                                                                    )

        train_state = train_state._replace(actor_state=train_state.actor_state.apply_gradients(grads=grads))

        # update target network
        new_critic_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
                                   lambda critic_state: critic_state.replace(
                                       target_params=optax.incremental_update(train_state.critic_state.params,
                                                                                      train_state.critic_state.target_params,
                                                                                      self.agent_config.TAU)),
                                   lambda train_state: train_state, operand=train_state.critic_state)
        new_actor_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
                                   lambda actor_state: actor_state.replace(
                                       target_params=optax.incremental_update(train_state.actor_state.params,
                                                                              train_state.actor_state.target_params,
                                                                              self.agent_config.TAU)),
                                   lambda train_state: train_state, operand=train_state.actor_state)
        train_state = train_state._replace(critic_state=new_critic_state)
        train_state = train_state._replace(actor_state=new_actor_state)
        # TODO is the above okay? think needs reworking if possible

        info = {"value_loss": jnp.mean(critic_loss),  # TODO technically q loss?
                "actor_loss": jnp.mean(actor_loss)
                }

        return train_state, mem_state, env_state, info, key