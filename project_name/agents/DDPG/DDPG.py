import sys
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax.training.train_state import TrainState
from project_name.utils import MemoryState, flip_and_switch
from project_name.agents import AgentBase
from project_name.agents.DDPG import get_DDPG_config, ContinuousQNetwork, DeterministicPolicy
import flashbax as fbx
import optax
from functools import partial
from typing import Any, NamedTuple
import flax
from flax.core.frozen_dict import freeze
import rlax
from project_name.utils import TransitionFlashbax, TrainStateExt
import chex


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

        self.config.NUM_EPISODES = self.config.TOTAL_TIMESTEPS // (self.agent_config.NUM_INNER_STEPS * self.config.NUM_ENVS)

        self.action_dim = env.action_space().shape[0]  # TODO check this

        self.critic_network = ContinuousQNetwork(config=config)
        self.actor_network = DeterministicPolicy(self.action_dim, env.action_space().low, env.action_space().high)

        key, _key = jrandom.split(key)

        init_x = (jnp.zeros((1, *env.observation_space(env_params).shape)),
                 (jnp.zeros((1, self.action_dim))))

        init_actor_x = jnp.zeros((1, *env.observation_space(env_params).shape))

        self.critic_network_params = self.critic_network.init(_key, *init_x)
        self.actor_network_params = self.actor_network.init(_key, init_actor_x)

        self.buffer = fbx.make_flat_buffer(max_length=self.agent_config.BUFFER_SIZE,
                                           min_length=self.agent_config.BATCH_SIZE,
                                           sample_batch_size=self.agent_config.BATCH_SIZE,
                                           add_sequences=True,
                                           add_batch_size=self.config.NUM_ENVS)

        self.buffer = self.buffer.replace(init=jax.jit(self.buffer.init),
                                          add=jax.jit(self.buffer.add, donate_argnums=0),
                                          sample=jax.jit(self.buffer.sample),
                                          can_sample=jax.jit(self.buffer.can_sample))

        # self.eps_scheduler = optax.linear_schedule(init_value=1.0,
        #                                            end_value=1e-6,
        #                                            transition_steps=self.agent_config.EPS_DECAY * config.NUM_EPISODES,
        #                                            )
        self.eps_scheduler = optax.exponential_decay(init_value=1.0,
                                                   end_value=1e-6,
                                                   decay_rate=self.agent_config.EPS_DECAY,
                                                   transition_steps=config.NUM_EPISODES,
                                                   )

        self.ACTION_SCALE = (env.action_space().high - env.action_space().low) / 2

    def create_train_state(self):
        actor_optim = optax.chain(
            optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM),
            optax.adam(self.agent_config.LR_ACTOR, eps=1e-5),
        )
        q_optim = optax.chain(
            optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM),
            optax.adam(self.agent_config.LR_CRITIC, eps=1e-5),
        )


        return (TrainStateDDPG(critic_state=TrainStateExt.create(apply_fn=self.critic_network.apply,
                                                                      params=self.critic_network_params,
                                                                      target_params=self.critic_network_params,
                                                                      tx=q_optim),
                              actor_state=TrainStateExt.create(apply_fn=self.actor_network.apply,
                                                               params=self.actor_network_params,
                                                               target_params=self.actor_network_params,
                                                               tx=actor_optim)),
                self.buffer.init(
                    TransitionFlashbax(done=jnp.zeros((), dtype=bool),
                                  action=jnp.zeros((self.action_dim,), dtype=jnp.float64),
                                  reward=jnp.zeros(()),
                                  obs=jnp.zeros((*self.env.observation_space(self.env_params).shape,),
                                                dtype=jnp.float32)
                                  )))

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):
        action = train_state.actor_state.apply_fn(train_state.actor_state.params, ac_in[0]).mode()
        key, _key = jrandom.split(key)

        # TODO check action can go to the max and min, it seems to be stuck at -0.6 and 0.6 currently maybe

        # def callback(action, state):
        #     print(action)
        #     print(state)
        #     print("NEW ONE")
        #
        # jax.experimental.io_callback(callback, None, action, ac_in[0])

        exploration_noise = self.eps_scheduler(train_state.n_updates)

        action = rlax.add_gaussian_noise(_key,
                                         action,
                                         self.ACTION_SCALE * exploration_noise).clip(self.env.action_space().low, self.env.action_space().high)

        return mem_state, action, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch_LNZ, unused_2):
        train_state, mem_state, ac_in, key = runner_state

        mem_state = self.buffer.add(mem_state, TransitionFlashbax(done=flip_and_switch(traj_batch_LNZ.done),
                                                                 action=flip_and_switch(traj_batch_LNZ.action),
                                                                 reward=flip_and_switch(traj_batch_LNZ.reward),
                                                                 obs=flip_and_switch(traj_batch_LNZ.obs),
                                                                 ))

        key, _key = jrandom.split(key)
        batch = self.buffer.sample(mem_state, _key)

        def critic_loss(critic_target_params, critic_params, actor_target_params, batch):
            obs_BO = batch.experience.first.obs
            action_BA = batch.experience.first.action
            reward_B = batch.experience.first.reward
            done_B = batch.experience.first.done
            nobs_BO = batch.experience.second.obs

            # action_pred_BA = train_state.actor_state.apply_fn(actor_target_params, nobs_BO).mode()
            # action_pred_BA = jnp.clip(action_pred_BA, self.env.action_space().low, self.env.action_space().high)
            # target_val_B1 = train_state.critic_state.apply_fn(critic_target_params, nobs_BO, action_pred_BA)
            #
            # y_expected_B = reward_B + (1 - done_B) * self.agent_config.GAMMA * jnp.squeeze(target_val_B1, axis=-1)  # TODO do I need stop gradient?
            #
            # y_pred_B1 = train_state.critic_state.apply_fn(critic_params, obs_BO, action_BA)
            #
            # # jax.debug.print("Q-values mean: {}", jnp.mean(y_pred_B1))
            # # jax.debug.print("Q-values std: {}", jnp.std(y_pred_B1))
            #
            # loss_critic = optax.losses.huber_loss(jnp.squeeze(y_pred_B1, axis=-1), y_expected_B) / 1.0  # same as smooth l1 loss ?

            q_tm1 = train_state.critic_state.apply_fn(critic_params, obs_BO, action_BA)  # TODO add shapes
            next_action = train_state.actor_state.apply_fn(actor_target_params, nobs_BO).mode().clip(self.env.action_space().low, self.env.action_space().high)
            q_t = train_state.critic_state.apply_fn(critic_target_params, nobs_BO, next_action)

            # Cast and clip rewards.
            discount = 1.0 - done_B.astype(jnp.float32)
            d_t = (discount * self.agent_config.GAMMA).astype(jnp.float32)
            # r_t = jnp.clip(reward_B, -config.system.max_abs_reward, config.system.max_abs_reward).astype(jnp.float32)

            def td_learning(v_tm1: chex.Array, r_t: chex.Array, discount_t: chex.Array, v_t: chex.Array, huber_loss_parameter: chex.Array) -> chex.Array:
                """Calculates the temporal difference error. Each input is a batch."""
                target_tm1 = r_t + discount_t * v_t
                td_errors = target_tm1 - v_tm1
                if huber_loss_parameter > 0.0:
                    batch_loss = rlax.huber_loss(td_errors, huber_loss_parameter)
                else:
                    batch_loss = rlax.l2_loss(td_errors)
                return jnp.mean(batch_loss)

            q_loss = td_learning(q_tm1, reward_B, d_t, q_t, self.agent_config.HUBER_LOSS_PARAM)

            return q_loss  # jnp.mean(loss_critic)

        critic_loss, grads = jax.value_and_grad(critic_loss, argnums=1, has_aux=False)(train_state.critic_state.target_params,
                                                                                      train_state.critic_state.params,
                                                                                      train_state.actor_state.target_params,
                                                                                      batch
                                                                                      )

        # critic_grad_norm = optax.global_norm(grads)
        # jax.debug.print("Critic gradient norm: {}", critic_grad_norm)

        new_critic_state = train_state.critic_state.apply_gradients(grads=grads)
        # train_state = train_state._replace(critic_state=train_state.critic_state.apply_gradients(grads=grads))

        def policy_loss(critic_params, actor_params, batch):
            obs_BO = batch.experience.first.obs

            action_pred_BA = train_state.actor_state.apply_fn(actor_params, obs_BO).mode().clip(self.env.action_space().low, self.env.action_space().high)
            q_val_B1 = train_state.critic_state.apply_fn(critic_params, obs_BO, action_pred_BA)
            loss_actor = -jnp.mean(q_val_B1)

            # def critic_mean(critic_params, obs, dpg_a_t):
            #     logits = train_state.critic_state.apply_fn(critic_params, obs, dpg_a_t)
            #
            #     return jnp.mean(logits)
            #
            # dpg_a_t_BA = action_pred_BA
            # dq_da_B1 = jax.vmap(jax.grad(critic_mean, argnums=2), in_axes=(None, 0, 0))(critic_params, obs_BO, dpg_a_t_BA)
            # dqda_clipping = None  # can also be 1
            # rlax_loss = jax.vmap(rlax.dpg_loss, in_axes=(0, 0, None))(dpg_a_t_BA, dq_da_B1, dqda_clipping)
            # rlax_loss = jnp.mean(rlax_loss)
            #
            # return rlax_loss
            return loss_actor

        actor_loss, grads = jax.value_and_grad(policy_loss, argnums=1, has_aux=False)(new_critic_state.params,
                                                                                    train_state.actor_state.params,
                                                                                    batch
                                                                                    )

        # actor_grad_norm = optax.global_norm(grads)
        # jax.debug.print("Actor gradient norm: {}", actor_grad_norm)

        new_actor_state = train_state.actor_state.apply_gradients(grads=grads)
        # train_state = train_state._replace(actor_state=train_state.actor_state.apply_gradients(grads=grads))

        # update target network
        # new_critic_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
        #                                 lambda new_critic_state:
        #                                 new_critic_state.replace(target_params=optax.incremental_update(new_critic_state.params,
        #                                                             new_critic_state.target_params,
        #                                                             self.agent_config.TAU)),
        #                                 lambda new_critic_state: new_critic_state, operand=new_critic_state)
        # new_actor_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
        #                                 lambda new_actor_state:
        #                                 new_actor_state.replace(target_params=optax.incremental_update(new_actor_state.params,
        #                                                             new_actor_state.target_params,
        #                                                             self.agent_config.TAU)),
        #                                 lambda new_actor_state: new_actor_state, operand=new_actor_state)
        # TODO above needs reworking if possible

        train_state = train_state._replace(critic_state=new_critic_state,
                                           actor_state=new_actor_state,
                                           n_updates=train_state.n_updates+1)

        info = {"value_loss": jnp.mean(critic_loss),
                "policy_loss": jnp.mean(actor_loss),
                "exploration_schedule": jnp.mean(self.eps_scheduler(train_state.n_updates))
                }

        return train_state, mem_state, info, key