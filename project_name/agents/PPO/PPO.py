import sys
import jax
import jax.numpy as jnp
from typing import Any
import jax.random as jrandom
from functools import partial
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState
from project_name.agents import AgentBase
from project_name.agents.PPO import get_PPO_config, DiscreteActorCritic, ContinuousActorCritic


class PPOAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_PPO_config()
        self.env = env
        self.env_params = env_params
        self.utils = utils

        self.config.NUM_EPISODES = self.config.TOTAL_TIMESTEPS // (self.agent_config.NUM_INNER_STEPS * self.config.NUM_ENVS)

        if self.config.DISCRETE:
            self.action_choices = env.action_space().n
            self.action_dim = 1
            self.network = DiscreteActorCritic(self.action_choices, config=config)
        else:
            self.action_choices = None
            self.action_dim = env.action_space().shape[0]  # TODO check this
            self.network = ContinuousActorCritic(self.action_dim, env.action_space().low, env.action_space().high)

        init_x = jnp.zeros((1, config.NUM_ENVS, *env.observation_space(env_params).shape))

        key, _key = jrandom.split(key)
        self.network_params = self.network.init(_key, init_x)

        self.agent_config.NUM_MINIBATCHES = min(self.config.NUM_ENVS, self.agent_config.NUM_MINIBATCHES)

        def linear_schedule(count):  # TODO put this somewhere better
            frac = (1.0 - (count // (self.agent_config.NUM_MINIBATCHES * self.agent_config.UPDATE_EPOCHS)) / config.NUM_UPDATES)
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
        return (TrainState.create(apply_fn=self.network.apply,
                                  params=self.network_params,
                                  tx=self.tx),
                MemoryState(hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
                            extras={
                                "values": jnp.zeros(self.config.NUM_ENVS, dtype=jnp.float32),
                                "log_probs": jnp.zeros(self.config.NUM_ENVS, dtype=jnp.float32),
                            })
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        mem_state = mem_state._replace(extras={"values": jnp.zeros(self.config.NUM_ENVS),
                                               "log_probs": jnp.zeros(self.config.NUM_ENVS),},
                                       hstate=jnp.zeros((self.config.NUM_ENVS, 1))
                                       )
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):
        pi, value, action_logits = train_state.apply_fn(train_state.params, ac_in[0])
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)

        # def callback(action):
        #     print(action)
        #     print("NEW ONE")
        #
        # jax.experimental.io_callback(callback, None, action_logits)

        # action = jnp.clip(action, self.env.action_space().low, self.env.action_space().high)  # TODO add for continuous somehow

        log_prob = pi.log_prob(action)

        mem_state.extras["values"] = value
        mem_state.extras["log_probs"] = log_prob

        new_mem_state = mem_state._replace(extras=mem_state.extras)

        return new_mem_state, action, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch_LNZ, unused_2):
        train_state, mem_state, ac_in, key = runner_state
        _, last_val, _ = train_state.apply_fn(train_state.params, ac_in[0])

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae_N, next_value_N = gae_and_next_value
                done_N, reward_N = (transition.done, transition.reward)
                value_N = transition.mem_state.extras["values"]
                delta_N = reward_N + self.agent_config.GAMMA * next_value_N * (1 - done_N) - value_N
                gae_N = (delta_N + self.agent_config.GAMMA * self.agent_config.GAE_LAMBDA * (1 - done_N) * gae_N)
                return (gae_N, value_N), gae_N

            _, advantages_LN = jax.lax.scan(_get_advantages,
                                         (jnp.zeros_like(last_val, dtype=jnp.float64), last_val),
                                         traj_batch,
                                         reverse=True,
                                         unroll=16,
                                         )
            return advantages_LN, advantages_LN + traj_batch.mem_state.extras["values"]

        advantages_LN, targets_LN = _calculate_gae(traj_batch_LNZ, last_val)

        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets, key = batch_info

                def _loss_fn(params, traj_batch, gae, targets, key):
                    # RERUN NETWORK
                    pi, value, _ = train_state.apply_fn(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.mem_state.extras["values"] + (value - traj_batch.mem_state.extras["values"]).clip(-self.agent_config.CLIP_EPS,
                                                                                            self.agent_config.CLIP_EPS)
                    value_losses = jnp.square(value - targets)

                    # def callback(action_logits):
                    #     print(action_logits)
                    #     print("NEW ONE")
                    #
                    # jax.experimental.io_callback(callback, None, value_losses)

                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(
                        where=(1 - traj_batch.done))

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.mem_state.extras["log_probs"])
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (jnp.clip(ratio,
                                            1.0 - self.agent_config.CLIP_EPS,
                                            1.0 + self.agent_config.CLIP_EPS,
                                            ) * gae)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean(where=(1 - traj_batch.done))
                    key, _key = jrandom.split(key)
                    entropy = pi.entropy(seed=_key).mean(where=(1 - traj_batch.done))

                    total_loss = (loss_actor
                                  + self.agent_config.VF_COEF * value_loss
                                  - self.agent_config.ENT_COEF * entropy
                                  )

                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets, key)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, key = update_state
            key, _key = jrandom.split(key)

            permutation = jrandom.permutation(_key, self.config.NUM_ENVS)
            batch = (traj_batch, advantages, targets)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(
                jnp.reshape(x, [x.shape[0], self.agent_config.NUM_MINIBATCHES, -1] + list(x.shape[2:]), ), 1, 0, ),
                                                 shuffled_batch)

            batch_key = jrandom.split(key, self.agent_config.NUM_MINIBATCHES)
            train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, (*minibatches, batch_key))

            update_state = (train_state,
                            traj_batch,
                            advantages,
                            targets,
                            key,
                            )
            return update_state, total_loss

        update_state = (train_state, traj_batch_LNZ, advantages_LN, targets_LN, key)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, self.agent_config.UPDATE_EPOCHS)
        train_state, traj_batch, advantages, targets, key = update_state

        info = {"value_loss": jnp.mean(loss_info[1][0]),
                "policy_loss": jnp.mean(loss_info[1][1]),
                "entropy": jnp.mean(loss_info[1][2]),
                }

        return train_state, mem_state, info, key
