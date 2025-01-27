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
from project_name.agents.IDQN import get_IDQN_config, RNNQNetwork, ScannedRNN
import flashbax as fbx


class IDQNTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class TransitionIDQN(NamedTuple):  # TODO can we standardise this for all dqn approaches?
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray


class IDQNAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_IDQN_config()
        self.env = env
        self.env_params = env_params
        self.utils = utils
        self.network = RNNQNetwork(env.action_space().n, config=config)  # TODO separate RNN and normal

        key, _key = jrandom.split(key)
        init_hstate = ScannedRNN.initialize_carry(config.NUM_ENVS, self.agent_config.GRU_HIDDEN_DIM)

        if self.config.CNN:
            init_x = ((jnp.zeros((1, config.NUM_ENVS, *env.observation_space(env_params).shape))),
                      jnp.zeros((1, config.NUM_ENVS)),
                      )
        else:
            init_x = (jnp.zeros((1, config.NUM_ENVS, utils.observation_space(env, env_params))),
                      jnp.zeros((1, config.NUM_ENVS)),
                      )

        self.network_params = self.network.init(_key, init_hstate, init_x)
        self.init_hstate = ScannedRNN.initialize_carry(config.NUM_ENVS,
                                                       self.agent_config.GRU_HIDDEN_DIM)  # TODO do we need both?

        self.agent_config.NUM_MINIBATCHES = min(self.config.NUM_ENVS, self.agent_config.NUM_MINIBATCHES)

        self.buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=self.agent_config.BUFFER_SIZE // config.NUM_ENVS,
            min_length_time_axis=self.agent_config.BATCH_SIZE,
            sample_batch_size=self.agent_config.BATCH_SIZE,
            add_batch_size=config.NUM_ENVS,
            sample_sequence_length=1,
            period=1,
        )

        self.eps_scheduler = optax.linear_schedule(init_value=self.agent_config.EPS_START,
                                                   end_value=self.agent_config.EPS_FINISH,
                                                   transition_steps=self.agent_config.EPS_DECAY * config.NUM_UPDATES,
                                                   )

        def linear_schedule(count):  # TODO put this somewhere better
            frac = (1.0 - (count // (
                    self.agent_config.NUM_MINIBATCHES * self.agent_config.UPDATE_EPOCHS)) / config.NUM_UPDATES)
            return self.agent_config.LR * frac

        if self.agent_config.ANNEAL_LR:
            self.tx = optax.chain(optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM),
                                  optax.adam(learning_rate=linear_schedule, eps=self.agent_config.ADAM_EPS),
                                  )
        else:
            self.tx = optax.chain(optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM),
                                  optax.adam(self.agent_config.LR, eps=self.agent_config.ADAM_EPS),
                                  )

    def create_train_state(self):
        return (IDQNTrainState.create(apply_fn=self.network.apply,
                                      params=self.network_params,
                                      target_network_params=self.network_params,
                                      tx=self.tx),
                self.buffer.init(TransitionIDQN(done=jnp.zeros((self.config.NUM_INNER_STEPS,), dtype=bool),
                                                action=jnp.zeros((self.config.NUM_INNER_STEPS,),
                                                                 dtype=jnp.int32),
                                                reward=jnp.zeros((self.config.NUM_INNER_STEPS,)),
                                                obs=jnp.zeros((self.config.NUM_INNER_STEPS,
                                                               self.utils.observation_space(self.env, self.env_params)),
                                                              dtype=jnp.int8),
                                                # TODO is it always an int for the obs?
                                                ))
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def _get_greedy_actions(self, q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def _eps_greedy_exploration(self, key, q_vals, eps, valid_actions):

        key, _key = jax.random.split(key)
        greedy_actions = self._get_greedy_actions(q_vals, valid_actions)

        # pick random actions from the valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _keys = jax.random.split(key, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_keys, valid_actions)

        chosen_actions = jnp.where(jax.random.uniform(_key, greedy_actions.shape) < eps,  # pick random actions
                                   random_actions,
                                   greedy_actions,
                                   )
        return chosen_actions

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        _, q_vals = train_state.apply_fn(train_state.params, None, ac_in)  # TODO no rnn for now
        eps = self.eps_scheduler(train_state.n_updates)
        q_vals = q_vals.squeeze(axis=0)
        valid_actions = jnp.ones_like(q_vals)
        action = self._eps_greedy_exploration(key, q_vals, eps, valid_actions)

        log_prob = jnp.zeros((1,))
        value = jnp.zeros((1,))  # TODO sort these out

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch, unused_2):
        traj_batch = jax.tree_map(lambda x: x[:, agent], traj_batch)
        # print(traj_batch)
        train_state, mem_state, env_state, ac_in, key = runner_state

        def flip_and_switch(tracer):
            return jnp.swapaxes(tracer, 0, 1)[:, jnp.newaxis]

        # buffer_traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1)[:, jnp.newaxis], traj_batch)

        mem_state = self.buffer.add(mem_state,
                                    TransitionIDQN(done=flip_and_switch(traj_batch.done),  # TODO check the dims here
                                                   action=flip_and_switch(traj_batch.action),
                                                   reward=flip_and_switch(traj_batch.reward),
                                                   obs=flip_and_switch(traj_batch.obs),
                                                   ))

        # NETWORKS UPDATE
        def _learn_phase(carry, _):
            train_state, key = carry
            key, _key = jax.random.split(key)
            minibatch = self.buffer.sample(mem_state, key).experience
            minibatch = jax.tree_map(lambda x: jnp.swapaxes(x[:, 0], 0, 1),
                                     minibatch)  # (max_time_steps, batch_size, ...)

            # # preprocess network input
            # init_hs = ScannedRNN.initialize_carry(
            #     config["HIDDEN_SIZE"],
            #     len(env.agents),
            #     config["BUFFER_BATCH_SIZE"],
            # )
            # timesteps, batch_size, ...
            _obs = minibatch.obs
            _dones = minibatch.done
            _actions = minibatch.action
            _rewards = minibatch.reward  # TODO sort these out at some point when working

            _, q_next_target = train_state.apply_fn(train_state.target_network_params, None,
                                                    (_obs, _dones))  # TODO no rnn for now

            def _loss_fn(params):
                _, q_vals = train_state.apply_fn(params, None, (_obs, _dones))  # TODO no rnn for now
                # (timesteps, batch_size, num_actions)

                # get logits of the chosen actions
                chosen_action_q_vals = jnp.take_along_axis(q_vals, _actions[..., jnp.newaxis], axis=-1).squeeze(-1)
                # TODO check the above is correct indexing
                # (timesteps, batch_size,)

                # unavailable_actions = 1 - _avail_actions
                valid_q_vals = q_vals  # - (unavailable_actions * 1e10)

                # get the q values of the next state
                q_next = jnp.take_along_axis(q_next_target, jnp.argmax(valid_q_vals, axis=-1)[..., jnp.newaxis],
                                             axis=-1).squeeze(-1)
                # (timesteps, batch_size,)

                target = (_rewards[:-1] + (1 - _dones[:-1]) * self.agent_config.GAMMA * q_next[1:])

                chosen_action_q_vals = chosen_action_q_vals[:-1]
                loss = jnp.mean((chosen_action_q_vals - jax.lax.stop_gradient(target)) ** 2)

                return loss, chosen_action_q_vals.mean()

            (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)

            return (train_state, key), (loss, qvals)

        key, _key = jax.random.split(key)
        is_learn_time = (self.buffer.can_sample(mem_state)) & (
                train_state.timesteps > self.agent_config.LEARNING_STARTS)
        (train_state, key), (loss, qvals) = jax.lax.cond(is_learn_time,
                                                         lambda train_state, key: jax.lax.scan(_learn_phase,
                                                                                               (train_state, key), None,
                                                                                               self.agent_config.UPDATE_EPOCHS),
                                                         lambda train_state, key: ((train_state, key),
                                                                                   (jnp.zeros(
                                                                                       self.agent_config.UPDATE_EPOCHS),
                                                                                    jnp.zeros(
                                                                                        self.agent_config.UPDATE_EPOCHS))),
                                                         train_state,
                                                         _key,
                                                         )

        # update target network
        train_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
                                   lambda train_state: train_state.replace(
                                       target_network_params=optax.incremental_update(train_state.params,
                                                                                      train_state.target_network_params,
                                                                                      self.agent_config.TAU)),
                                   lambda train_state: train_state, operand=train_state)

        info = {"value_loss": jnp.mean(loss),  # TODO technically q loss?
                }

        return train_state, mem_state, env_state, info, key
