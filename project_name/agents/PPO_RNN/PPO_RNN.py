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
from project_name.agents.PPO_RNN import get_PPORNN_config, ActorCriticRNN, ScannedRNN


class PPO_RNNAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_PPORNN_config()
        self.env = env
        self.network = ActorCriticRNN(env.action_space().n, config=config, agent_config=self.agent_config)

        key, _key = jrandom.split(key)
        init_hstate = ScannedRNN.initialize_carry(config.NUM_ENVS, self.agent_config.GRU_HIDDEN_DIM)

        # if self.config.CNN:
        #     init_x = ((jnp.zeros((1, config.NUM_ENVS, *utils.observation_space(env_params)["observation"].shape)),
        #                jnp.zeros((1, config.NUM_ENVS, utils.observation_space(env_params)["inventory"].shape))),
        #               jnp.zeros((1, config.NUM_ENVS)),
        #               )
        # else:
        init_x = (jnp.zeros((1, config.NUM_ENVS, *utils.observation_space(env, env_params))),
                  jnp.zeros((1, config.NUM_ENVS)),
                  )

        self.network_params = self.network.init(_key, init_hstate, init_x)
        self.init_hstate = ScannedRNN.initialize_carry(config.NUM_ENVS,
                                                       self.agent_config.GRU_HIDDEN_DIM)  # TODO do we need both?

        self.agent_config.NUM_MINIBATCHES = min(self.config.NUM_ENVS, self.agent_config.NUM_MINIBATCHES)

        def linear_schedule(count):  # TODO put this somewhere better and think this is right?
            frac = (1.0 - (count // (self.agent_config.NUM_MINIBATCHES * self.agent_config.UPDATE_EPOCHS)) / config.NUM_UPDATES)
            # frac = 1 - count // 16 / num_updates
            return self.agent_config.LR * frac

        if self.agent_config.ANNEAL_LR:
            self.tx = optax.chain(optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM),
                                  optax.adam(learning_rate=linear_schedule, eps=1e-5),
                                  )
        else:
            self.tx = optax.chain(optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM),
                                  optax.adam(self.agent_config.LR, eps=1e-5),
                                  )

    def create_train_state(self):
        return (TrainState.create(apply_fn=self.network.apply,
                                  params=self.network_params,
                                  tx=self.tx),
                MemoryState(hstate=self.init_hstate,
                            extras={
                                "action_logits": jnp.zeros((self.config.NUM_ENVS, 1, self.env.action_space().n)),
                                "values": jnp.zeros((self.config.NUM_ENVS, 1)),
                                "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
                            }, ),
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        mem_state = mem_state._replace(extras={
            "action_logits": jnp.zeros((self.config.NUM_ENVS, 1, self.env.action_space().n)),
            "values": jnp.zeros((self.config.NUM_ENVS, 1)),
            "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
        },
            hstate=jnp.zeros((self.config.NUM_ENVS, self.agent_config.GRU_HIDDEN_DIM)),
        )
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        hstate, pi, value, action_logits = train_state.apply_fn(train_state.params, mem_state.hstate, ac_in)
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        # sets shape as num_envs, 1 (idk what it is), the rest
        mem_state.extras["action_logits"] = jnp.swapaxes(action_logits, 0, 1)  # TODO check the right dimensions here
        mem_state.extras["values"] = jnp.swapaxes(value, 0, 1)
        mem_state.extras["log_probs"] = jnp.swapaxes(log_prob, 0, 1)  # TODO sort this out a bit

        mem_state = mem_state._replace(hstate=hstate, extras=mem_state.extras)

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch, all_mem_state):
        traj_batch = jax.tree_map(lambda x: x[:, agent], traj_batch)
        # CALCULATE ADVANTAGE
        train_state, mem_state, env_state, ac_in, key = runner_state
        # ac_in = (last_obs[jnp.newaxis, :],
        #          last_done[jnp.newaxis, :],
        #          )
        _, _, last_val, _ = train_state.apply_fn(train_state.params, mem_state.hstate, ac_in)
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
            def _update_minibatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    # RERUN NETWORK
                    _, pi, value, _ = train_state.apply_fn(params,
                                                      init_hstate.squeeze(axis=0),
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
                total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, init_hstate, traj_batch, advantages, targets, key = update_state
            key, _key = jrandom.split(key)

            # adding an additional "fake" dimensionality to perform minibatching correctly
            init_hstate = jnp.reshape(mem_state.hstate, (1, self.config.NUM_ENVS, -1))

            permutation = jrandom.permutation(_key, self.config.NUM_ENVS)
            # traj_batch shape = [num_inner_steps, num_envs, *more]
            batch = (init_hstate,  # TODO check this axis swapping etc if it works
                     traj_batch,
                     advantages,
                     targets)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(
                jnp.reshape(x, [x.shape[0], self.agent_config.NUM_MINIBATCHES, -1] + list(x.shape[2:]), ), 1, 0, ),
                                                 shuffled_batch)
            # shapes are [batches, 1, num_envs/batches, h_state], [batches, num_steps, num_envs/batches]
            train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)

            update_state = (train_state,
                            mem_state,
                            traj_batch,
                            advantages,
                            targets,
                            key,
                            )
            return update_state, total_loss

        update_state = (train_state, mem_state, traj_batch, advantages, targets, key)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, self.agent_config.UPDATE_EPOCHS)
        train_state, mem_state, traj_batch, advantages, targets, key = update_state

        info = {"value_loss": jnp.mean(loss_info[1][0]),
                "actor_loss": jnp.mean(loss_info[1][1]),
                "entropy": jnp.mean(loss_info[1][2]),
                }

        return train_state, mem_state, env_state, info, key
