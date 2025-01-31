import sys
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple
import jax.random as jrandom
from functools import partial
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState, TransitionFlashbax, flip_and_switch
from project_name.agents import AgentBase
import chex
from project_name.agents.BootDQN import get_BootDQN_config, EnsembleNetwork
import numpy as np
import distrax
import flax
import rlax
import flashbax as fbx


class TrainStateRP(TrainState):  # TODO check gradients do not update the static prior
    static_prior_params: flax.core.FrozenDict


class TrainStateBootDQN(NamedTuple):
    ensemble_state: TrainStateRP
    ensemble_target_state: TrainStateRP
    n_updates: int = 0


class TransitionBootDQN(NamedTuple):  # TODO can I extend this from TransitionFlasbax at some point and inheret?
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    noise: jnp.ndarray
    mask: jnp.ndarray


class BootDQNAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_BootDQN_config()
        self.env = env
        self.env_params = env_params
        self.utils = utils

        if self.config.DISCRETE:
            self.rp_network = EnsembleNetwork(env.action_space().n, config=config, agent_config=self.agent_config)
            self.action_dim = 1
            self.action_choices = env.action_space().n
        else:
            print("Q learning continuous no no")
            sys.exit()

        if self.config.CNN:
            self._init_x = jnp.zeros((1, config.NUM_ENVS, *env.observation_space(env_params).shape))
        else:
            self._init_x = jnp.zeros((1, config.NUM_ENVS, *utils.observation_space(env, env_params)))

        self.key = key

        self.tx = optax.adam(self.agent_config.LR)

        self.buffer = fbx.make_flat_buffer(max_length=self.agent_config.BUFFER_SIZE,
                                           min_length=self.agent_config.BATCH_SIZE,
                                           sample_batch_size=self.agent_config.BATCH_SIZE,
                                           add_sequences=True,
                                           add_batch_size=self.config.NUM_ENVS)  # TODO should this be the real batch size?

        self.buffer = self.buffer.replace(init=jax.jit(self.buffer.init),
                                          add=jax.jit(self.buffer.add, donate_argnums=0),
                                          sample=jax.jit(self.buffer.sample),
                                          can_sample=jax.jit(self.buffer.can_sample))

        self.eps_scheduler = optax.linear_schedule(init_value=self.agent_config.EPS_START,
                                                   end_value=self.agent_config.EPS_FINISH,
                                                   transition_steps=self.agent_config.EPS_DECAY * config.NUM_EPISODES,
                                                   )

    def create_train_state(self):
        def create_ensemble_state(key: chex.PRNGKey) -> TrainStateRP:  # TODO is this the best place to put it all?
            key, _key = jrandom.split(key)
            rp_params = self.rp_network.init(_key, self._init_x)["params"],
            reward_state = TrainStateRP.create(apply_fn=self.rp_network.apply,
                                             params=rp_params[0]["_net"],  # TODO unsure why it needs a 0 index here?
                                             static_prior_params=rp_params[0]["_prior_net"],
                                             tx=optax.adam(self.agent_config.ENS_LR))
            return reward_state

        ensemble_keys = jrandom.split(self.key, self.agent_config.NUM_ENSEMBLE)

        ensemble_train_state = jax.vmap(create_ensemble_state, in_axes=(0,))(ensemble_keys)

        return (TrainStateBootDQN(ensemble_train_state,
                                  ensemble_train_state),
                self.buffer.init(TransitionBootDQN(done=jnp.zeros((), dtype=bool),
                                 action = jnp.zeros((1,), dtype=jnp.int32),
                                 reward = jnp.zeros(()),
                                 obs = jnp.zeros((*self.utils.observation_space(self.env, self.env_params),), dtype=jnp.float32),
                                 noise = jnp.zeros((self.agent_config.NUM_ENSEMBLE,)),
                                 mask = jnp.zeros((self.agent_config.NUM_ENSEMBLE,)))
                                 )
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def _eps_greedy_exploration(self, key, q_vals_NA, eps):
        key, _key = jax.random.split(key)
        greedy_actions_N = jnp.argmax(q_vals_NA, axis=-1)
        random_actions_N = jrandom.randint(_key, greedy_actions_N.shape, 0, self.action_choices)

        key, _key = jax.random.split(key)
        chosen_actions_N = jnp.where(jax.random.uniform(_key, greedy_actions_N.shape) < eps,  # pick random actions
                                   random_actions_N,
                                   greedy_actions_N,
                                   )
        return chosen_actions_N

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: Any, key: Any):
        q_vals_UNA = jax.vmap(train_state.ensemble_state.apply_fn, in_axes=(0, None))({"params": {"_net": train_state.ensemble_state.params,
                                                      "_prior_net": train_state.ensemble_state.static_prior_params}}, ac_in[0])
        q_vals_NA = q_vals_UNA[0]  # TODO need to add thompson sampling random selection
        eps = self.eps_scheduler(train_state.n_updates)
        action_N = self._eps_greedy_exploration(key, q_vals_NA, eps)

        return mem_state, action_N, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward_noise(self, ens_state: TrainStateRP, obs_LNO: chex.Array, actions_LNA: chex.Array, key) -> chex.Array:
        ensemble_obs_ULNO = jnp.broadcast_to(obs_LNO, (self.agent_config.NUM_ENSEMBLE, *obs_LNO.shape))
        ensemble_action_ULNA = jnp.broadcast_to(actions_LNA, (self.agent_config.NUM_ENSEMBLE, *actions_LNA.shape))

        def single_reward_noise(ens_state: TrainStateRP, obs_LNO: chex.Array, action_LNA: chex.Array) -> chex.Array:
            rew_pred = ens_state.apply_fn({"params": {"_net": ens_state.params,
                                                      "_prior_net": ens_state.static_prior_params}},
                                          obs_LNO, action_LNA)
            return rew_pred

        ensembled_reward_ULN1 = jax.vmap(single_reward_noise)(ens_state,
                                                         ensemble_obs_ULNO,
                                                         ensemble_action_ULNA)

        ensembled_reward_LN1 = self.agent_config.UNCERTAINTY_SCALE * jnp.var(ensembled_reward_ULN1, axis=0)

        return ensembled_reward_LN1

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch_LNZ, unused_2):
        train_state, mem_state, env_state, ac_in, key = runner_state

        key, _key = jrandom.split(key)
        mask_LNU = jrandom.binomial(_key, 1, self.agent_config.MASK_PROB,
                                (*traj_batch_LNZ[0].shape, self.agent_config.NUM_ENSEMBLE))
        noise_LNU = jrandom.normal(_key, (*traj_batch_LNZ[0].shape, self.agent_config.NUM_ENSEMBLE))



        mem_state = self.buffer.add(mem_state, TransitionBootDQN(done=flip_and_switch(traj_batch_LNZ.done),
                                                                  action=flip_and_switch(jnp.expand_dims(traj_batch_LNZ.action, axis=-1)),
                                                                  reward=flip_and_switch(traj_batch_LNZ.reward),
                                                                  obs=flip_and_switch(traj_batch_LNZ.obs),
                                                                  noise=flip_and_switch(mask_LNU),
                                                                  mask=flip_and_switch(noise_LNU)
                                                                  ))

        key, _key = jrandom.split(key)
        batch = self.buffer.sample(mem_state, _key)

        def train_ensemble(ensemble_state, target_state, noise, mask, batch):
            def reward_predictor_loss(rp_params, prior_params, rp_target_params, prior_target_params, noise_B, mask_B, batch):
                obs_BO = batch.experience.first.obs
                action_BA = batch.experience.first.action
                reward_B = batch.experience.first.reward
                done_B = batch.experience.first.done
                nobs_BO = batch.experience.second.obs

                q_tm1_BC = ensemble_state.apply_fn({"params": {"_net": rp_params, "_prior_net": prior_params}}, obs_BO)
                q_t_BC = ensemble_state.apply_fn({"params": {"_net": rp_target_params, "_prior_net": prior_target_params}}, nobs_BO)

                reward_B += self.agent_config.REWARD_NOISE_SCALE * noise_B

                td_error_B = jax.vmap(rlax.q_learning)(q_tm1_BC, jnp.squeeze(action_BA, axis=-1), reward_B, self.agent_config.GAMMA * (1 - done_B), q_t_BC)

                # q_tm1 = network.apply(params, o_tm1)
                # q_t = network.apply(target_params, o_t)
                # r_t += noise_scale * z_t
                # batch_q_learning = jax.vmap(rlax.q_learning)
                # td_error = batch_q_learning(q_tm1, a_tm1, r_t, discount * d_t, q_t)
                # return jnp.mean(m_t * td_error ** 2)

                return 0.5 * jnp.mean(mask_B * jnp.square(td_error_B))

            ensemble_loss, grads = jax.value_and_grad(reward_predictor_loss, argnums=0)(ensemble_state.params,
                                                                                        ensemble_state.static_prior_params,
                                                                                        target_state.params,
                                                                                        target_state.static_prior_params,
                                                                                        noise,
                                                                                        mask,
                                                                                        batch)
            ensemble_state = ensemble_state.apply_gradients(grads=grads)

            new_target_state = jax.lax.cond(train_state.n_updates % self.agent_config.TARGET_UPDATE_INTERVAL == 0,
                                            lambda target_state:
                                            target_state.replace(
                                                params=optax.incremental_update(ensemble_state.params,
                                                                                       target_state.params,
                                                                                       self.agent_config.TAU)),
                                            lambda target_state: target_state, operand=target_state)

            return ensemble_loss, ensemble_state, new_target_state

        ind_noise = batch.experience.first.noise.T
        ind_mask = batch.experience.first.mask.T
        ensembled_loss, ens_state, new_target_state = jax.vmap(train_ensemble, in_axes=(0, 0, 0, 0, None))(train_state.ensemble_state,
                                                                                                train_state.ensemble_target_state,
                                                                                                ind_noise,
                                                                                                ind_mask,
                                                                                                batch)
        train_state = train_state._replace(ensemble_state=ens_state,
                                           ensemble_target_state=new_target_state,
                                           n_updates=train_state.n_updates+1)

        info = {"value_loss": jnp.mean(ensembled_loss),
                }
        for ensemble_id in range(self.agent_config.NUM_ENSEMBLE):
            info[f"Ensemble_{ensemble_id}_Loss"] = ensembled_loss[ensemble_id]

        return train_state, mem_state, env_state, info, key