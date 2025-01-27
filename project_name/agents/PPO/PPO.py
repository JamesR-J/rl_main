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
from project_name.agents.PPO import get_PPO_config, ActorCritic


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
        self.network = ActorCritic(env.action_space().n, config=config)

        if self.config.CNN:
            init_x = ((jnp.zeros((1, config.NUM_ENVS, *env.observation_space(env_params).shape))),
                      jnp.zeros((1, config.NUM_ENVS)),
                      )
        else:
            init_x = (jnp.zeros((1, config.NUM_ENVS, utils.observation_space(env, env_params))),
                      jnp.zeros((1, config.NUM_ENVS)),
                      )

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
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        pi, value, action_logits = train_state.apply_fn(train_state.params, ac_in)
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch, unused_2):
        traj_batch = jax.tree_map(lambda x: x[:, agent], traj_batch)
        # print(traj_batch)
        train_state, mem_state, env_state, ac_in, key = runner_state
        _, last_val, _ = train_state.apply_fn(train_state.params, ac_in)
        last_val = last_val.squeeze(axis=0)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + self.agent_config.GAMMA * next_value * (1 - done) - value
                gae = (delta + self.agent_config.GAMMA * self.agent_config.GAE_LAMBDA * (1 - done) * gae)
                return (gae, value), gae

            _, advantages = jax.lax.scan(_get_advantages,
                                         (jnp.zeros_like(last_val), last_val),
                                         traj_batch,
                                         reverse=True,
                                         unroll=16,
                                         )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value, _ = train_state.apply_fn(params,
                                                      (traj_batch.obs,
                                                       traj_batch.done,
                                                       # traj_batch.avail_actions
                                                       ),
                                                      )
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-self.agent_config.CLIP_EPS,
                                                                                            self.agent_config.CLIP_EPS)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(
                        where=(1 - traj_batch.done))

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (jnp.clip(ratio,
                                            1.0 - self.agent_config.CLIP_EPS,
                                            1.0 + self.agent_config.CLIP_EPS,
                                            ) * gae)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean(where=(1 - traj_batch.done))
                    entropy = pi.entropy().mean(where=(1 - traj_batch.done))

                    total_loss = (loss_actor
                                  + self.agent_config.VF_COEF * value_loss
                                  - self.agent_config.ENT_COEF * entropy
                                  )

                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, key = update_state
            key, _key = jrandom.split(key)

            permutation = jrandom.permutation(_key, self.config.NUM_ENVS)
            batch = (traj_batch,
                     advantages,
                     targets)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(
                jnp.reshape(x, [x.shape[0], self.agent_config.NUM_MINIBATCHES, -1] + list(x.shape[2:]), ), 1, 0, ),
                                                 shuffled_batch, )

            train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)

            update_state = (train_state,
                            traj_batch,
                            advantages,
                            targets,
                            key,
                            )
            return update_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, key)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, self.agent_config.UPDATE_EPOCHS)
        train_state, traj_batch, advantages, targets, key = update_state

        info = {"value_loss": jnp.mean(loss_info[1][0]),
                "actor_loss": jnp.mean(loss_info[1][1]),
                "entropy": jnp.mean(loss_info[1][2]),
                }

        return train_state, mem_state, env_state, info, key
