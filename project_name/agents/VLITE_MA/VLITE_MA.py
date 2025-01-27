import sys
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple, Tuple
import jax.random as jrandom
from functools import partial
import optax
from flax.training.train_state import TrainState
from project_name.utils import MemoryState
from project_name.agents import AgentBase
import chex
from project_name.agents.VLITE_MA import get_VLITEMA_config, ActorCritic, EnsembleNetwork, EnsembleOppNetwork, binomial
import numpy as np
import distrax
import flax
import rlax
from distrax._src.utils import math
from project_name.utils import remove_element, remove_element_2, remove_element_3


class TrainStateRP(TrainState):  # TODO check gradients do not update the static prior
    static_prior_params: flax.core.FrozenDict


class TrainStateVLITE(NamedTuple):
    ac_state: TrainState
    ens_state: TrainStateRP
    opp_state: TrainStateRP


class VLITE_MAAgent(AgentBase):
    def __init__(self,
                 env,
                 env_params,
                 key,
                 config,
                 utils):
        self.config = config
        self.agent_config = get_VLITEMA_config()
        self.env = env
        self.env_params = env_params
        self.network = ActorCritic(env.action_space().n, config=config, agent_config=self.agent_config)
        self.rp_network = EnsembleNetwork(config=config, agent_config=self.agent_config)
        self.opp_rp_network = EnsembleOppNetwork(env.action_space().n, config=config, agent_config=self.agent_config)

        if self.config.CNN:
            self._init_x = jnp.zeros((1, config.NUM_ENVS, *env.observation_space(env_params).shape))
        else:
            self._init_x = jnp.zeros((1, config.NUM_ENVS, utils.observation_space(env, env_params)))

        key, _key = jrandom.split(key)
        self.network_params = self.network.init(_key, self._init_x)

        self.key = key

        self.tx = optax.adam(self.agent_config.LR)

    def create_train_state(self):
        def create_ensemble_state(key: chex.PRNGKey) -> TrainStateRP:  # TODO is this the best place to put it all?
            key, _key = jrandom.split(key)
            rp_params = self.rp_network.init(_key,
                                             self._init_x,
                                             jnp.zeros((1, self.config.NUM_ENVS, 1)),
                                             jnp.zeros((1, self.config.NUM_ENVS, self.config.NUM_AGENTS - 1)), )[
                "params"],
            reward_state = TrainStateRP.create(apply_fn=self.rp_network.apply,
                                               params=rp_params[0]["_net"],  # TODO unsure why it needs a 0 index here?
                                               static_prior_params=rp_params[0]["_prior_net"],
                                               tx=optax.adam(self.agent_config.ENS_LR))
            return reward_state

        def create_opp_ensemble_state(key: chex.PRNGKey) -> TrainStateRP:
            key, _key = jrandom.split(key)
            rp_params = self.opp_rp_network.init(_key,
                                                 self._init_x,
                                                 jnp.zeros((1, self.config.NUM_ENVS, 1)))["params"],
            reward_state = TrainStateRP.create(apply_fn=self.opp_rp_network.apply,
                                               params=rp_params[0]["_net"],  # TODO unsure why it needs a 0 index here?
                                               static_prior_params=rp_params[0]["_prior_net"],
                                               tx=optax.adam(self.agent_config.OPP_ENS_LR))
            return reward_state

        ensemble_keys = jrandom.split(self.key, self.agent_config.NUM_ENSEMBLE)
        return (TrainStateVLITE(ac_state=TrainState.create(apply_fn=self.network.apply,
                                                           params=self.network_params,
                                                           tx=self.tx),
                                ens_state=jax.vmap(create_ensemble_state, in_axes=(0,))(ensemble_keys),
                                opp_state=jax.vmap(create_opp_ensemble_state, in_axes=(0,))(ensemble_keys)),
                MemoryState(hstate=jnp.zeros((self.config.NUM_ENVS, 1)),
                            extras={"values": jnp.zeros((self.config.NUM_ENVS, 1)),
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
    def act(self, train_state: TrainStateVLITE, mem_state: Any, ac_in: Any, key: Any):  # TODO better implement checks
        pi, value, action_logits = train_state.ac_state.apply_fn(train_state.ac_state.params, ac_in[0])
        key, _key = jrandom.split(key)
        action = pi.sample(seed=_key)
        log_prob = pi.log_prob(action)

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_opp_logits(self, ens_state: TrainStateRP, obs: chex.Array, ego_actions, key) -> Tuple[
        chex.Array, chex.Array]:
        def single_opp_logits(ens_state: TrainStateRP, obs: chex.Array, ego_actions, key) -> Tuple[
            chex.Array, chex.Array]:
            logits = ens_state.apply_fn({"params": {"_net": ens_state.params,
                                                    "_prior_net": ens_state.static_prior_params}},
                                        obs, jnp.expand_dims(ego_actions, axis=-1))
            key, _key = jrandom.split(key)
            rho = distrax.Categorical(logits=logits)
            action = rho.sample(seed=_key)

            return action, logits

        ens_key = jrandom.split(key, self.agent_config.NUM_ENSEMBLE)
        all_action, all_opp_logits = jax.vmap(single_opp_logits, in_axes=(0, None, None, 0))(ens_state, obs,
                                                                                             ego_actions, ens_key)

        all_action = self.agent_config.ACTION_UNCERTAINTY_SCALE * jnp.var(all_action, axis=0)
        # all_action = jnp.minimum(all_action, 1.0)

        return all_action, all_opp_logits

    @partial(jax.jit, static_argnums=(0,))
    def _opp_logits_over_actions(self, ens_state: TrainStateRP, obs: chex.Array, key) -> Tuple[chex.Array, chex.Array]:
        # run the get_reward_noise for each action choice, can probs vmap
        ego_actions = jnp.expand_dims(jnp.arange(0, self.env.action_space().n, step=1), axis=(-1, -2))
        ego_actions = jnp.broadcast_to(ego_actions, (ego_actions.shape[0], obs.shape[0], obs.shape[1]))

        action_over_actions, logits_over_actions = jax.vmap(self._get_opp_logits, in_axes=(None, None, 0, None))(ens_state, obs,
                                                                                               ego_actions, key)
        action_over_actions = jnp.moveaxis(action_over_actions, 0, 2)

        logits_over_actions = jnp.moveaxis(logits_over_actions, 0, 3)  # TODO check this is okay

        return action_over_actions, logits_over_actions

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward_noise(self, ens_state: TrainStateRP, obs: chex.Array, actions: chex.Array,
                          opp_actions: chex.Array) -> chex.Array:
        def single_reward_noise(ens_state: TrainStateRP, obs: chex.Array, action: chex.Array, opp_action) -> chex.Array:
            rew_pred = ens_state.apply_fn({"params": {"_net": ens_state.params,
                                                      "_prior_net": ens_state.static_prior_params}},
                                          obs, jnp.expand_dims(action, axis=-1), jnp.expand_dims(opp_action, axis=-1))
            return rew_pred

        ensembled_reward = jax.vmap(single_reward_noise, in_axes=(0, None, None, None))(ens_state,
                                                                                        obs,
                                                                                        actions,
                                                                                        opp_actions)

        ensembled_reward = self.agent_config.UNCERTAINTY_SCALE * jnp.std(ensembled_reward, axis=0)
        ensembled_reward = jnp.minimum(ensembled_reward, 1.0)

        return ensembled_reward

    @partial(jax.jit, static_argnums=(0,))
    def _reward_noise_over_actions(self, ens_state: TrainStateRP, obs: chex.Array) -> chex.Array:
        # run the get_reward_noise for each action choice, can probs vmap twice?
        actions = jnp.arange(0, self.env.action_space().n, step=1)
        actions = jnp.broadcast_to(jnp.expand_dims(actions, axis=(-2, -1)),
                                   (*actions.shape, obs.shape[0], self.config.NUM_ENVS))

        opp_actions = jnp.arange(0, self.env.action_space().n, step=1)
        opp_actions = jnp.broadcast_to(jnp.expand_dims(opp_actions, axis=(-2, -1)),
                                       (*opp_actions.shape, obs.shape[0], self.config.NUM_ENVS))

        # obs = jnp.broadcast_to(obs, (actions.shape[0], actions.shape[1], *obs.shape))

        reward_over_actions = jax.vmap(jax.vmap(self._get_reward_noise, in_axes=(None, None, None, 0)),
                                       in_axes=(None, None, 0, None))(ens_state, obs, actions, opp_actions)
        # TODO check the above aswell

        reward_over_actions = jnp.swapaxes(jnp.squeeze(reward_over_actions, axis=-1), 0, 2)
        reward_over_actions = jnp.swapaxes(reward_over_actions, 1, 3)  # TODO these should be correct way around

        return reward_over_actions

    @partial(jax.jit, static_argnums=(0,))
    def _entropy_loss_fn(self, logits_t_ego, logits_t_opp, uncertainty_t, opp_action_noise):
        # log_pi_ego = jax.nn.log_softmax(logits_t_ego)
        # pi_times_log_pi_ego = math.mul_exp(log_pi_ego, log_pi_ego)
        #
        # log_pi_opp = jax.nn.log_softmax(logits_t_opp)
        # pi_times_log_pi_opp = math.mul_exp(log_pi_opp, log_pi_opp)
        #
        # sigma_rho_term = jnp.einsum('sij,sj->si', uncertainty_t, pi_times_log_pi_opp)
        #
        # return -jnp.sum(pi_times_log_pi_ego * sigma_rho_term, axis=-1)

        logits_t_ego = jnp.expand_dims(logits_t_ego, axis=-1)
        pi_ego = jax.nn.softmax(logits_t_ego)
        pi_opp = jax.nn.softmax(logits_t_opp)
        joint_prob = pi_ego * pi_opp

        log_pi_ego = jax.nn.log_softmax(logits_t_ego)
        log_pi_opp = jax.nn.log_softmax(logits_t_opp)

        opp_action_noise = jnp.expand_dims(opp_action_noise, axis=-1)  # TODO unsure if right additional dim and sum?
        #
        # return -jnp.sum(uncertainty_t * joint_prob * (log_pi_ego + log_pi_opp), axis=(-2, -1))
        return -jnp.sum(uncertainty_t * opp_action_noise * joint_prob * (log_pi_ego + log_pi_opp), axis=(-2, -1))
        # return jnp.zeros((logits_t_ego.shape[0], logits_t_ego.shape[-1]))

    @partial(jax.jit, static_argnums=(0,))
    def update(self, runner_state, agent, traj_batch, all_mem_state):
        action_opp = remove_element_3(traj_batch.action, agent)  # TODO ensure this is correct innit

        # opp_policy = all_mem_state[1].extras["action_logits"]  # TODO only works for two agents for now! with VLITE being in position 0

        traj_batch = jax.tree_map(lambda x: x[:, agent], traj_batch)
        train_state, mem_state, env_state, ac_in, key = runner_state

        obs = jnp.concatenate((traj_batch.obs, jnp.zeros((1, *traj_batch.obs.shape[1:]))), axis=0)
        # above is for the on policy version

        state_action_reward_noise = self._get_reward_noise(train_state.ens_state, traj_batch.obs, traj_batch.action,
                                                           action_opp)
        state_reward_noise = self._reward_noise_over_actions(train_state.ens_state, traj_batch.obs)
        opp_action_noise, opp_logits_all = self._opp_logits_over_actions(train_state.opp_state, traj_batch.obs, key)
        _, opp_logits = self._get_opp_logits(train_state.opp_state, traj_batch.obs, traj_batch.action, key)
        key, _key = jrandom.split(key)
        opp_int = jrandom.randint(_key, (1,), 0, self.agent_config.NUM_ENSEMBLE - 1)  # thompson sample?
        opp_logits = jnp.squeeze(opp_logits.at[opp_int].get(), axis=0)  # TODO check this
        opp_logits_all = jnp.squeeze(opp_logits_all.at[opp_int].get(), axis=0)   # TODO check this
        # opp_logits = jnp.expand_dims(opp_logits[0], axis=2)
        # opp_logits = opp_policy

        def ac_loss(params, opp_logits, opp_logits_all, trajectory, obs, state_action_reward_noise, opp_action_noise, action_opp):
            # TODO should this be a joint critic?
            _, values, logits = train_state.ac_state.apply_fn(params, obs)
            policy_dist = distrax.Categorical(logits=logits[:-1])  # ensure this is the same as the network distro
            log_prob = policy_dist.log_prob(trajectory.action)

            td_lambda = jax.vmap(rlax.td_lambda, in_axes=(1, 1, 1, 1, None), out_axes=1)
            k_estimate = td_lambda(values[:-1],
                                   trajectory.reward + (jnp.squeeze(state_action_reward_noise, axis=-1)),
                                   (1 - trajectory.done) * self.agent_config.GAMMA,
                                   values[1:],
                                   self.agent_config.TD_LAMBDA,
                                   )

            value_loss = jnp.mean(jnp.square(values[:-1] - jax.lax.stop_gradient(k_estimate)))
            # TODO this be the same since we are using values instead of qs and implicitly covers all actions?

            entropy = jax.vmap(self._entropy_loss_fn, in_axes=1, out_axes=1)(logits[:-1],
                                                                             jax.lax.stop_gradient(opp_logits_all),
                                                                             jax.lax.stop_gradient(state_reward_noise),
                                                                             jax.lax.stop_gradient(opp_action_noise))
            # TODO probably be good to add a mask to ensure don't do entropy on end steps

            opp_policy_dist = distrax.Categorical(logits=opp_logits)
            opp_log_prob = opp_policy_dist.log_prob(action_opp)

            policy_loss = -jnp.mean((log_prob + jax.lax.stop_gradient(opp_log_prob)) * jax.lax.stop_gradient(
                k_estimate - values[:-1]) + entropy)
            # policy_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(k_estimate - values[:-1]) + entropy)

            total_loss = policy_loss + value_loss
            # total_loss = value_loss  # TODO have changed this for now from the above

            return total_loss, entropy

        (pv_loss, entropy), grads = jax.value_and_grad(ac_loss, has_aux=True, argnums=0)(train_state.ac_state.params,
                                                                                         opp_logits,
                                                                                         opp_logits_all,
                                                                                         traj_batch,
                                                                                         obs,
                                                                                         state_action_reward_noise,
                                                                                         opp_action_noise,
                                                                                         action_opp)
        train_state = train_state._replace(ac_state=train_state.ac_state.apply_gradients(grads=grads))

        # train ensemble
        def train_ensemble(ens_state: TrainStateRP, obs, actions, opp_actions, rewards, mask):
            def reward_predictor_loss(rp_params, prior_params, obs, actions, opp_actions, rewards, mask):
                rew_pred = ens_state.apply_fn({"params": {"_net": rp_params,
                                                          "_prior_net": prior_params}},
                                              obs, jnp.expand_dims(actions, axis=-1),
                                              jnp.expand_dims(opp_actions, axis=-1))
                # rew_pred += reward_noise_scale * jnp.expand_dims(z_t, axis=-1)
                return 0.5 * jnp.mean(mask * jnp.square(jnp.squeeze(rew_pred, axis=-1) - rewards)), rew_pred
                # return jnp.mean(jnp.zeros((2))), rew_pred

            (ensemble_loss, rew_pred), grads = jax.value_and_grad(reward_predictor_loss, argnums=0, has_aux=True)(
                ens_state.params,
                ens_state.static_prior_params,
                obs,
                actions,
                opp_actions,
                rewards,
                mask)
            ens_state = ens_state.apply_gradients(grads=grads)

            return ensemble_loss, ens_state, rew_pred

        key, _key = jrandom.split(key)
        ensemble_mask = binomial(_key, 1, self.agent_config.MASK_PROB, (self.agent_config.NUM_ENSEMBLE,
                                                                        *entropy.shape))

        ensembled_loss, ens_state, rew_pred = jax.vmap(train_ensemble, in_axes=(0, None, None, None, None, 0))(
            train_state.ens_state,
            traj_batch.obs,
            traj_batch.action,
            action_opp,
            traj_batch.reward,
            ensemble_mask)
        train_state = train_state._replace(ens_state=ens_state)

        # train opp ensemble
        def train_opp_ensemble(opp_state: TrainStateRP, obs, ego_actions, actions, mask, key):
            def action_predictor_loss(rp_params, prior_params, obs, ego_actions, actions, mask, key):
                logit_pred = opp_state.apply_fn({"params": {"_net": rp_params, "_prior_net": prior_params}},
                                                obs, jnp.expand_dims(ego_actions, axis=-1))
                key, _key = jrandom.split(key)
                rho = distrax.Categorical(logits=logit_pred)
                action_pred = rho.sample(seed=_key)
                # rew_pred += reward_noise_scale * jnp.expand_dims(z_t, axis=-1)

                categorical_cross_entropy = optax.softmax_cross_entropy_with_integer_labels(logit_pred, actions)

                return jnp.mean(mask * categorical_cross_entropy), action_pred
                # return jnp.mean(jnp.zeros((2,))), rew_pred

            (opp_ens_loss, action_pred), grads = jax.value_and_grad(action_predictor_loss, argnums=0, has_aux=True)(
                opp_state.params,
                opp_state.static_prior_params,
                obs,
                ego_actions,
                actions,
                mask,
                key)
            opp_state = opp_state.apply_gradients(grads=grads)

            return opp_ens_loss, opp_state, action_pred

        ensemble_mask = np.random.binomial(1, self.agent_config.MASK_PROB, (self.agent_config.NUM_ENSEMBLE,
                                                                            *entropy.shape))  # ensure this is okay
        ens_key = jrandom.split(key, self.agent_config.NUM_ENSEMBLE)
        opp_ens_loss, opp_state, action_pred = jax.vmap(train_opp_ensemble, in_axes=(0, None, None, None, 0, 0))(
            train_state.opp_state,
            traj_batch.obs,
            traj_batch.action,
            action_opp,
            ensemble_mask,
            ens_key)
        train_state = train_state._replace(opp_state=opp_state)

        info = {"ac_loss": pv_loss,
                "entropy": jnp.mean(entropy),
                }
        for ensemble_id in range(self.agent_config.NUM_ENSEMBLE):
            info[f"Ensemble_{ensemble_id}_Reward_Pred_pv"] = rew_pred[ensemble_id, 6, 6]  # index random step/batch
            info[f"Ensemble_{ensemble_id}_Loss"] = ensembled_loss[ensemble_id]
            info[f"Opp_Ensemble_{ensemble_id}_Action_Pred"] = action_pred[ensemble_id, 6, 6]  # index random step/batch
            info[f"Opp_Ensemble_{ensemble_id}_Loss"] = opp_ens_loss[ensemble_id]

        return train_state, mem_state, env_state, info, key
