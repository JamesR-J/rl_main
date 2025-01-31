import sys
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple
import jax.random as jrandom
from functools import partial
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState, flip_and_switch
from project_name.agents import AgentBase
import chex
from project_name.agents.ERSAC import get_ERSAC_config, DiscreteActorCritic, ContinuousActorCritic, DiscreteEnsembleNetwork, ContinuousEnsembleNetwork
import numpy as np
import distrax
import flax
import rlax
import flashbax as fbx


class TrainStateERSAC(NamedTuple):
    ac_state: TrainState
    ens_state: Any  # TODO how to update this?
    log_tau: Any
    tau_opt_state: Any


class TrainStateRP(TrainState):  # TODO check gradients do not update the static prior
    static_prior_params: flax.core.FrozenDict


class TransitionERSAC(NamedTuple):  # TODO can I extend this from TransitionFlasbax at some point and inheret?
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    noise: jnp.ndarray
    mask: jnp.ndarray


class ERSACAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_ERSAC_config()
        self.env = env
        self.env_params = env_params
        self.utils = utils

        if self.config.DISCRETE:
            self.action_choices = env.action_space().n
            self.action_dim = 1
            self.network = DiscreteActorCritic(self.action_choices, config=config, agent_config=self.agent_config)
            self.rp_network = DiscreteEnsembleNetwork(self.action_choices, config=config, agent_config=self.agent_config)
            self.rp_init_x = jnp.zeros((1, self.config.NUM_ENVS))
        else:
            self.action_choices = None
            self.action_dim = env.action_space().shape[0]  # TODO check this
            self.network = ContinuousActorCritic(self.action_dim, env.action_space().low, env.action_space().high)
            self.rp_network = ContinuousEnsembleNetwork(self.action_dim, config=config, agent_config=self.agent_config)
            self.rp_init_x = jnp.zeros((1, self.config.NUM_ENVS, self.action_dim))

        if self.config.CNN:
            self._init_x = jnp.zeros((1, config.NUM_ENVS, *env.observation_space(env_params).shape))
        else:
            self._init_x = jnp.zeros((1, config.NUM_ENVS, *env.observation_space(env_params).shape))

        key, _key = jrandom.split(key)
        self.network_params = self.network.init(_key, self._init_x)

        self.log_tau = jnp.asarray(jnp.log(self.agent_config.INIT_TAU), dtype=jnp.float32)
        # self.log_tau = jnp.asarray(self.agent_config.INIT_TAU, dtype=jnp.float32)
        self.tau_optimiser = optax.adam(learning_rate=self.agent_config.TAU_LR)

        self.key = key

        self.tx = optax.adam(self.agent_config.LR)

        self.buffer = fbx.make_trajectory_buffer(add_batch_size=self.config.NUM_ENVS,  # TODO make it prioritised
                                                             sample_batch_size=self.agent_config.BATCH_SIZE,
                                                             sample_sequence_length=self.agent_config.SAMPLE_SEQ_LENGTH,
                                                             period=self.agent_config.SAMPLE_SEQ_LENGTH,
                                                             min_length_time_axis=self.agent_config.SAMPLE_SEQ_LENGTH,
                                                             max_size=self.agent_config.BUFFER_SIZE,
                                                             # priority_exponent=0.6,
                                                             # device=self.config.DEVICE
                                                 )

        self.buffer = self.buffer.replace(init=jax.jit(self.buffer.init),
                                          add=jax.jit(self.buffer.add, donate_argnums=0),
                                          sample=jax.jit(self.buffer.sample),
                                          can_sample=jax.jit(self.buffer.can_sample))

    def create_train_state(self):
        def create_ensemble_state(key: chex.PRNGKey) -> TrainState:  # TODO is this the best place to put it all?
            key, _key = jrandom.split(key)
            rp_params = self.rp_network.init(_key, self._init_x, self.rp_init_x)["params"],
            reward_state = TrainStateRP.create(apply_fn=self.rp_network.apply,
                                             params=rp_params[0]["_net"],  # TODO unsure why it needs a 0 index here?
                                             static_prior_params=rp_params[0]["_prior_net"],
                                             tx=optax.adam(self.agent_config.ENS_LR))
            return reward_state

        ensemble_keys = jrandom.split(self.key, self.agent_config.NUM_ENSEMBLE)
        if self.config.DISCRETE:
            action_init = jnp.zeros((), dtype=self.env.action_space().dtype)
        else:
            action_init = jnp.zeros((self.action_dim,), dtype=self.env.action_space().dtype)
        return (TrainStateERSAC(ac_state=TrainState.create(apply_fn=self.network.apply,
                                                           params=self.network_params,
                                                           tx=self.tx),
                                ens_state=jax.vmap(create_ensemble_state, in_axes=(0))(ensemble_keys),
                                log_tau=self.log_tau,
                                tau_opt_state=self.tau_optimiser.init(self.log_tau)),
                self.buffer.init(
                    TransitionERSAC(done=jnp.zeros((), dtype=bool),
                                       action=action_init,
                                       reward=jnp.zeros(()),
                                       obs=jnp.zeros((*self.env.observation_space(self.env_params).shape,),
                                                     dtype=jnp.float32),
                                       noise = jnp.zeros((self.agent_config.NUM_ENSEMBLE,)),
                                       mask = jnp.zeros((self.agent_config.NUM_ENSEMBLE,)))
                                       )
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: TrainStateERSAC, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        pi, value_N, action_logits_NA = train_state.ac_state.apply_fn(train_state.ac_state.params, ac_in[0])
        key, _key = jrandom.split(key)
        action_NA = pi.sample(seed=_key)
        # log_prob_NA = pi.log_prob(action_NA)
        #
        # mem_state.extras["values"] = value_N
        # mem_state.extras["log_probs"] = log_prob_NA

        # action = jnp.ones_like(action)  # TODO for testing randomized actions

        return mem_state, action_NA, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward_noise(self, ens_state: TrainStateRP, batch, key) -> chex.Array:
        def single_reward_noise(ens_state: TrainStateRP, obs_BSO: chex.Array, action_BSA: chex.Array) -> chex.Array:
            rew_pred_BS1 = ens_state.apply_fn({"params": {"_net": ens_state.params,
                                                      "_prior_net": ens_state.static_prior_params}},
                                          obs_BSO, action_BSA)  # TODO needs a 1 if only 1 dimensional action
            return rew_pred_BS1

        obs_BSO = batch.experience.obs
        actions_BSA = batch.experience.action

        ensembled_reward_UBS = jax.vmap(single_reward_noise, in_axes=(0, None, None))(ens_state, obs_BSO, actions_BSA)

        ensembled_reward_BS = self.agent_config.UNCERTAINTY_SCALE * jnp.var(ensembled_reward_UBS, axis=0)

        return ensembled_reward_BS

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch_LNZ, unused_2):
        train_state, mem_state, ac_in, key = runner_state

        key, _key = jrandom.split(key)
        mask_LNU = jrandom.binomial(_key, 1, self.agent_config.MASK_PROB,
                                    (*traj_batch_LNZ[0].shape, self.agent_config.NUM_ENSEMBLE))
        noise_LNU = jrandom.normal(_key, (*traj_batch_LNZ[0].shape, self.agent_config.NUM_ENSEMBLE))

        mem_state = self.buffer.add(mem_state, TransitionERSAC(done=flip_and_switch(traj_batch_LNZ.done),
                                                               action=flip_and_switch(traj_batch_LNZ.action),
                                                                reward=flip_and_switch(traj_batch_LNZ.reward),
                                                                obs=flip_and_switch(traj_batch_LNZ.obs),
                                                               noise = flip_and_switch(noise_LNU),
                                                               mask=flip_and_switch(mask_LNU)
                                                               )
                                    )

        key, _key = jrandom.split(key)
        batch_BSZ = self.buffer.sample(mem_state, _key)

        # obs_LP1NO = jnp.concatenate((traj_batch_LNZ.obs, jnp.zeros((1, *traj_batch_LNZ.obs.shape[1:]))), axis=0)

        state_action_reward_noise_BS = self._get_reward_noise(train_state.ens_state, batch_BSZ, key)

        def ac_loss(params, batch_BSZ, tau_params, state_action_reward_noise_BS, key):
            tau = jnp.exp(tau_params)

            obs_BSO = batch_BSZ.experience.obs
            action_BSA = batch_BSZ.experience.action
            reward_BS = batch_BSZ.experience.reward.astype(jnp.float32)  # TODO some dodgy dtype need to sort out
            done_BS = batch_BSZ.experience.done

            pi, values_BS, _ = train_state.ac_state.apply_fn(params, obs_BSO)
            # policy_dist = distrax.Categorical(logits=logitsLP1N2[:-1])  # ensure this is the same as the network distro
            # log_prob_LN = policy_dist.log_prob(trajectory_LNZ.action)
            log_prob_BS = pi.log_prob(action_BSA)

            td_lambda = jax.vmap(rlax.td_lambda, in_axes=(0, 0, 0, 0, None))
            k_estimate_LN = td_lambda(values_BS[:, :-1],
                                      reward_BS[:, :-1] + (state_action_reward_noise_BS[:, :-1] / (2 * tau)),
                                      (1 - done_BS[:, :-1]) * self.agent_config.GAMMA,
                                      values_BS[:, 1:],
                                      self.agent_config.TD_LAMBDA,
                                      )

            value_loss = jnp.mean(jnp.square(values_BS[:, :-1] - jax.lax.stop_gradient(k_estimate_LN - tau * log_prob_BS[:, :-1])))
            # TODO is it right to use [1:] for these values etc or [:-1]?

            key, _key = jrandom.split(key)
            # entropy_BS = pi.entropy(seed=_key)
            entropy_BS = pi.entropy()  # TODO between discrete and continuous no key is needed

            policy_loss = -jnp.mean(log_prob_BS[:, :-1] * jax.lax.stop_gradient(k_estimate_LN - values_BS[:, :-1]) + tau * entropy_BS[:, :-1])

            # _, values_LP1N, logitsLP1N2 = train_state.ac_state.apply_fn(params, obs_LP1NO)
            # policy_dist = distrax.Categorical(logits=logitsLP1N2[:-1])  # ensure this is the same as the network distro
            # log_prob_LN = policy_dist.log_prob(trajectory_LNZ.action)
            #
            # td_lambda = jax.vmap(rlax.td_lambda, in_axes=(1, 1, 1, 1, None), out_axes=1)
            # k_estimate_LN = td_lambda(values_LP1N[:-1],
            #                        trajectory_LNZ.reward + (jnp.squeeze(state_action_reward_noise_LN1, axis=-1) / (2 * tau)),
            #                        (1 - trajectory_LNZ.done) * self.agent_config.GAMMA,
            #                        values_LP1N[1:],
            #                        self.agent_config.TD_LAMBDA,
            #                        )
            #
            # value_loss = jnp.mean(jnp.square(values_LP1N[:-1] - jax.lax.stop_gradient(k_estimate_LN - tau * log_prob_LN)))
            # # TODO is it right to use [1:] for these values etc or [:-1]?
            #
            # entropy = policy_dist.entropy()
            #
            # policy_loss = -jnp.mean(log_prob_LN * jax.lax.stop_gradient(k_estimate_LN - values_LP1N[:-1]) + tau * entropy)

            return policy_loss + value_loss, (policy_loss, value_loss, entropy_BS)

        (pv_loss, loss_info), grads = jax.value_and_grad(ac_loss, has_aux=True, argnums=0)(train_state.ac_state.params,
                                                                                         batch_BSZ,
                                                                                         train_state.log_tau,
                                                                                         state_action_reward_noise_BS,
                                                                                         key)
        train_state = train_state._replace(ac_state=train_state.ac_state.apply_gradients(grads=grads))

        def tau_loss(tau_params, entropy_BS, state_action_reward_noise_BS):
            tau = jnp.exp(tau_params)

            tau_loss = state_action_reward_noise_BS / (2 * tau) + (tau * entropy_BS)

            return jnp.mean(tau_loss)

        tau_loss_val, tau_grads = jax.value_and_grad(tau_loss, has_aux=False, argnums=0)(train_state.log_tau,
                                                                                         loss_info[2],
                                                                                         state_action_reward_noise_BS)
        tau_updates, new_tau_opt_state = self.tau_optimiser.update(tau_grads, train_state.tau_opt_state)
        new_tau_params = optax.apply_updates(train_state.log_tau, tau_updates)
        train_state = train_state._replace(log_tau=new_tau_params, tau_opt_state=new_tau_opt_state)

        def train_ensemble(ens_state: TrainStateRP, noise_BS, mask_BS, batch_BSZ):
            def reward_predictor_loss(rp_params, prior_params, noise_BS, mask_BS, batch_BSZ):
                obs_BSO = batch_BSZ.experience.obs
                action_BSA = batch_BSZ.experience.action
                reward_BS = batch_BSZ.experience.reward.astype(jnp.float32)  # TODO some dodgy dtype need to sort out

                rew_pred_BS = ens_state.apply_fn({"params": {"_net": rp_params, "_prior_net": prior_params}}, obs_BSO, action_BSA)
                rew_pred_BS += self.agent_config.REWARD_NOISE_SCALE * noise_BS
                return 0.5 * jnp.mean(mask_BS * jnp.square(rew_pred_BS - reward_BS)), rew_pred_BS
                # return jnp.mean(jnp.zeros((2))), rew_pred

            (ensemble_loss, rew_pred), grads = jax.value_and_grad(reward_predictor_loss, argnums=0, has_aux=True)(ens_state.params,
                                                                                                                  ens_state.static_prior_params,
                                                                                                                  noise_BS,
                                                                                                                  mask_BS,
                                                                                                                  batch_BSZ)
            ens_state = ens_state.apply_gradients(grads=grads)

            return ensemble_loss, ens_state, rew_pred

        ind_noise_UBS = jnp.moveaxis(batch_BSZ.experience.noise, [-3, -2, -1], [-2, -1, -3])
        ind_mask_UBS = jnp.moveaxis(batch_BSZ.experience.mask, [-3, -2, -1], [-2, -1, -3])

        ensembled_loss, ens_state, rew_pred = jax.vmap(train_ensemble, in_axes=(0, 0, 0, None))(train_state.ens_state,
                                                                       ind_noise_UBS,
                                                                       ind_mask_UBS,
                                                                       batch_BSZ)
        train_state = train_state._replace(ens_state=ens_state)

        info = {"value_loss": jnp.mean(loss_info[0]),
                "policy_loss": jnp.mean(loss_info[1]),
                "entropy": jnp.mean(loss_info[2]),
                "tau_loss": tau_loss_val,
                "avg_ensemble_loss": jnp.mean(ensembled_loss),
                "tau": jnp.exp(new_tau_params),
                }
        for ensemble_id in range(self.agent_config.NUM_ENSEMBLE):
            # info[f"Ensemble_{ensemble_id}_Reward_Pred_pv"] = rew_pred[ensemble_id, 6, 6]  # index random step and random batch
            info[f"Ensemble_{ensemble_id}_Reward_Pred_pv"] = jnp.mean(rew_pred[ensemble_id])  # index random step and random batch
            info[f"Ensemble_{ensemble_id}_Loss"] = ensembled_loss[ensemble_id]

        return train_state, mem_state, info, key
