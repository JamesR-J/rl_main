import sys
import os
import importlib
import jax.numpy as jnp
from typing import NamedTuple, Any, Mapping
import chex
import jax
from flax.training.train_state import TrainState
import flax


class MemoryState(NamedTuple):
    hstate: jnp.ndarray
    extras: Mapping[str, jnp.ndarray]


class TrainStateExt(TrainState):
    target_params: flax.core.FrozenDict


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mem_state: MemoryState
    # env_state: Any  # TODO added this but can change
    info: jnp.ndarray


class EvalTransition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    distribution: Any
    spec_key: chex.PRNGKey
    env_state: jnp.ndarray


def import_class_from_folder(folder_name):
    """
    Imports a class from a folder with the same name

    Args:
        folder_name (str): The name of the folder and potential class.

    Returns:
        The imported class, or None if import fails.
    """

    if not isinstance(folder_name, str):
        raise TypeError("folder_name must be a string.")

    # Check for multiple potential entries
    potential_path = os.path.join(os.curdir, "project_name", "agents",
                                  folder_name)  # TODO the project_name addition ain't great

    if os.path.isdir(potential_path) and os.path.exists(
            os.path.join(potential_path, f"{folder_name}.py")):
        # Use importlib to dynamically import the module
        module_spec = importlib.util.spec_from_file_location(folder_name,
                                                             os.path.join(potential_path, f"{folder_name}.py"))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        # Retrieve the class from the imported module
        return getattr(module, f"{folder_name}Agent")

    else:
        print(f"Error: Folder '{folder_name}' not found in any search paths.")
        return None


def ipd_visitation(traj_batch: Transition, final_obs: jnp.ndarray) -> dict:  # TODO make this n player version at some point
    observations = traj_batch.obs[:, 0, :][:, jnp.newaxis, :]  # TODO index to agent 0 again
    actions = traj_batch.action[:, 0, :][:, jnp.newaxis, :]  # TODO index to agent 0 again
    final_obs = final_obs[0]  # TODO index to agent 0

    # obs [num_inner_steps, num_agents, num_envs, ...]
    # final_t [num_opps, num_envs, ...]
    num_timesteps = observations.shape[0]
    # obs = [0, 1, 2, 3, 4], a = [0, 1]
    # combine = [0, .... 9]
    state_actions = 2 * jnp.argmax(observations, axis=-1) + actions
    state_actions = jnp.reshape(state_actions, (num_timesteps,) + state_actions.shape[1:], )
    # assume final step taken is cooperate

    final_obs = jax.lax.expand_dims(2 * jnp.argmax(final_obs, axis=-1), [0])
    state_actions = jnp.append(state_actions, final_obs[jnp.newaxis, :, :], axis=0)
    hist = jnp.bincount(state_actions.flatten(), length=10)
    state_freq = hist.reshape((int(hist.shape[0] / 2), 2)).sum(axis=1)
    state_probs = state_freq / state_freq.sum()
    action_probs = jnp.nan_to_num(hist[::2] / state_freq)
    return {
        "state_visitation/CC": state_freq[0],
        "state_visitation/CD": state_freq[1],
        "state_visitation/DC": state_freq[2],
        "state_visitation/DD": state_freq[3],
        "state_visitation/START": state_freq[4],
        "state_probability/CC": state_probs[0],
        "state_probability/CD": state_probs[1],
        "state_probability/DC": state_probs[2],
        "state_probability/DD": state_probs[3],
        "state_probability/START": state_probs[4],
        "cooperation_probability/CC": action_probs[0],
        "cooperation_probability/CD": action_probs[1],
        "cooperation_probability/DC": action_probs[2],
        "cooperation_probability/DD": action_probs[3],
        "cooperation_probability/START": action_probs[4],
    }


def cg_visitation(state: Any) -> dict:  # TODO sort this to work
    # [num_opps, num_envs, num_outer_episodes]  # actually is [num_envs, num_outer_episodes]
    total_1 = state.red_coop + state.red_defect
    total_2 = state.blue_coop + state.blue_defect
    avg_prob_1 = jnp.sum(state.red_coop, axis=-1) / jnp.sum(total_1, axis=-1)
    avg_prob_2 = jnp.sum(state.blue_coop, axis=-1) / jnp.sum(total_2, axis=-1)
    final_prob_1 = state.red_coop[:, -1] / total_1[:, -1]
    final_prob_2 = state.blue_coop[:, -1] / total_2[:, -1]

    # [num_opps, num_envs, num_states]  # actually is [num_envs, num_states
    prob_coop_1 = jnp.sum(state.coop1, axis=(0,)) / jnp.sum(state.counter, axis=(0,))
    prob_coop_2 = jnp.sum(state.coop2, axis=(0,)) / jnp.sum(state.counter, axis=(0,))
    count = jnp.nanmean(state.counter, axis=(0,))
    return {
        "prob_coop/1": jnp.nanmean(avg_prob_1),  # [1]
        "prob_coop/2": jnp.nanmean(avg_prob_2),  # [1]
        "final_prob_coop/1": jnp.nanmean(final_prob_1),  # [1]
        "final_prob_coop/2": jnp.nanmean(final_prob_2),  # [1]
        "total_coins/1": total_1.sum(),  # int
        "total_coins/2": total_2.sum(),  # int
        "coins_per_episode/1": total_1.sum(axis=-1).mean(axis=(0,)),  # [1]
        "coins_per_episode/2": total_2.sum(axis=-1).mean(axis=(0,)),  # [1]
        "final_coin_total/1": total_1[:, -1].mean(axis=(0,)),  # [1]
        "final_coin_total/2": total_2[:, -1].mean(axis=(0,)),  # [1]
        "cooperation_probability/1/SS": prob_coop_1[0],
        "cooperation_probability/1/CC": prob_coop_1[1],
        "cooperation_probability/1/CD": prob_coop_1[2],
        "cooperation_probability/1/DC": prob_coop_1[3],
        "cooperation_probability/1/DD": prob_coop_1[4],
        "cooperation_probability/1/SC": prob_coop_1[5],
        "cooperation_probability/1/SD": prob_coop_1[6],
        "cooperation_probability/1/CS": prob_coop_1[7],
        "cooperation_probability/1/DS": prob_coop_1[8],
        "cooperation_probability/2/SS": prob_coop_2[0],
        "cooperation_probability/2/CC": prob_coop_2[1],
        "cooperation_probability/2/CD": prob_coop_2[2],
        "cooperation_probability/2/DC": prob_coop_2[3],
        "cooperation_probability/2/DD": prob_coop_2[4],
        "cooperation_probability/2/SC": prob_coop_2[5],
        "cooperation_probability/2/SD": prob_coop_2[6],
        "cooperation_probability/2/CS": prob_coop_2[7],
        "cooperation_probability/2/DS": prob_coop_2[8],
        "state_visitation/SS": count[0],
        "state_visitation/CC": count[1],
        "state_visitation/CD": count[2],
        "state_visitation/DC": count[3],
        "state_visitation/DD": count[4],
        "state_visitation/SC": count[5],
        "state_visitation/SD": count[6],
        "state_visitation/CS": count[7],
        "state_visitation/DS": count[8],
    }


def ipditm_stats(state: Any, traj_batch: Transition, num_envs: int) -> dict:
    from .pax.envs.in_the_matrix import Actions

    traj1_actions = traj_batch.action[:, 0, :][:, jnp.newaxis, :]
    traj2_actions = traj_batch.action[:, 1, :][:, jnp.newaxis, :]
    traj1_rewards = traj_batch.reward[:, 0, :][:, jnp.newaxis, :]
    traj2_rewards = traj_batch.reward[:, 1, :][:, jnp.newaxis, :]
    traj1_obs = traj_batch.obs[1][:, 0, :][:, jnp.newaxis, :]
    traj2_obs = traj_batch.obs[1][:, 1, :][:, jnp.newaxis, :]

    """Compute statistics for IPDITM."""
    interacts1 = (
        jnp.count_nonzero(traj1_actions == Actions.interact) / num_envs
    )
    interacts2 = (
        jnp.count_nonzero(traj2_actions == Actions.interact) / num_envs
    )

    soft_reset_mask = jnp.where(traj1_rewards != 0, 1, 0)
    num_soft_resets = jnp.count_nonzero(traj1_rewards) / num_envs

    num_sft_resets = jnp.maximum(1, num_soft_resets)
    coops1 = (
        soft_reset_mask * traj1_obs[..., 0]
    ).sum() / (num_envs * num_sft_resets)
    defect1 = (
        soft_reset_mask * traj1_obs[..., 1]
    ).sum() / (num_envs * num_sft_resets)
    coops2 = (
        soft_reset_mask * traj2_obs[..., 0]
    ).sum() / (num_envs * num_sft_resets)
    defect2 = (
        soft_reset_mask * traj2_obs[..., 1]
    ).sum() / (num_envs * num_sft_resets)

    rewards1 = traj1_rewards.sum() / num_envs
    rewards2 = traj2_rewards.sum() / num_envs
    f_rewards1 = traj1_rewards[-1, ...].sum() / num_envs
    f_rewards2 = traj2_rewards[-1, ...].sum() / num_envs

    return {
        "interactions/1": interacts1,
        "interactions/2": interacts2,
        "coop_coin/1": coops1,
        "coop_coin/2": coops2,
        "defect_coin/1": defect1,
        "defect_coin/2": defect2,
        "total_coin/1": coops1 + defect1,
        "total_coin/2": coops2 + defect2,
        "ratio/1": jnp.nan_to_num(coops1 / (coops1 + defect1), nan=0),
        "ratio/2": jnp.nan_to_num(coops2 / (coops2 + defect2), nan=0),
        "num_soft_resets": num_soft_resets,
        "train/total_reward/player_1": rewards1,
        "train/total_reward/player_2": rewards2,
        "train/final_reward/player1": f_rewards1,
        "train/final_reward/player2": f_rewards2,
    }


def remove_element(arr, index):  # TODO can improve?
    if arr.shape[-1] == 1:
        raise ValueError("Cannot remove element from an array of size 1")
    elif arr.shape[-1] == 2:
        return jnp.expand_dims(arr[:, :, 1 - index], -1)
    else:
        return jnp.concatenate([arr[:, :, :index], arr[:, :, index + 1:]])


def remove_element_2(arr, index):  # TODO can improve?
    if arr.shape[-2] == 1:
        raise ValueError("Cannot remove element from an array of size 1")
    elif arr.shape[-2] == 2:
        return jnp.expand_dims(arr[:, :, 1 - index, :], -2)
    else:
        return jnp.concatenate([arr[:, :, :index, :], arr[:, :, index + 1:, :]])


def remove_element_3(arr, index):  # TODO can improve?
    if arr.shape[-2] == 1:
        raise ValueError("Cannot remove element from an array of size 1")
    elif arr.shape[-2] == 2:
        return arr[:, 1 - index, :]
    else:
        return jnp.concatenate([arr[:, :index, :], arr[:, index + 1:, :]])


class Utils_IMG:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def batchify(x: dict, agent_list, num_agents, num_envs):
        inter = jnp.stack([x[a] for a in agent_list])
        return inter.reshape((num_agents, num_envs, -1))

    @staticmethod
    def batchify_obs(x: dict, agent_list, num_agents, num_envs):
        inter = jnp.stack([x[a] for a in agent_list])
        return inter.reshape((num_agents, num_envs, -1))

    @staticmethod
    def unbatchify(x: jnp.ndarray, agent_list, num_agents, num_devices):
        x = x.reshape((num_agents, num_devices, -1))
        return {i: x[i] for i in agent_list}

    @staticmethod
    def ac_in(obs, dones, agent):
        return (obs[jnp.newaxis, agent, :],
                dones[jnp.newaxis, agent],
                )

    @staticmethod
    def visitation(env_state, traj_batch, final_obs):
        return ipd_visitation(traj_batch, final_obs)

    @staticmethod
    def observation_space(env, env_params):
        return env.observation_space(env_params).n


class Utils_CG(Utils_IMG):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def visitation(env_state, traj_batch, final_obs):
        return cg_visitation(env_state)

    @staticmethod
    def observation_space(env, env_params):
        return env.observation_space(env_params).shape[0]


class Utils_IMPITM(Utils_IMG):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def batchify_obs(x: dict, agent_list, num_agents, num_envs):
        # obs = jnp.stack([x[a]["observation"] for a in agent_list]).reshape(
        #     (num_agents, num_envs, *x[0]["observation"].shape[1:]))
        # inv = jnp.stack([x[a]["inventory"] for a in agent_list]).reshape((num_agents, num_envs, -1))
        # return (obs, inv)
        inter = jnp.stack([x[a] for a in agent_list])
        return inter.reshape((num_agents, num_envs, *inter.shape[2:]))

    @staticmethod
    def ac_in(obs, dones, agent):
        # return ((obs[0][jnp.newaxis, agent, :],
        #          obs[1][jnp.newaxis, agent, :]),
        #         dones[jnp.newaxis, agent],
        #         )
        return (obs[jnp.newaxis, agent, :],
                dones[jnp.newaxis, agent],
                )

    def visitation(self, env_state, traj_batch, final_obs):
        return ipditm_stats(env_state, traj_batch, self.config.NUM_ENVS)


class Utils_DEEPSEA(Utils_IMG):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def batchify_obs(x: dict, agent_list, num_agents, num_envs):
        # obs = jnp.stack([x[a]["observation"] for a in agent_list]).reshape(
        #     (num_agents, num_envs, *x[0]["observation"].shape[1:]))
        # inv = jnp.stack([x[a]["inventory"] for a in agent_list]).reshape((num_agents, num_envs, -1))
        # return (obs, inv)
        inter = jnp.stack([x[a] for a in agent_list])
        return inter.reshape((num_agents, num_envs, *inter.shape[2:]))

    @staticmethod
    def ac_in(obs, dones, agent):
        # return ((obs[0][jnp.newaxis, agent, :],
        #          obs[1][jnp.newaxis, agent, :]),
        #         dones[jnp.newaxis, agent],
        #         )
        return (obs[jnp.newaxis, agent, :],
                dones[jnp.newaxis, agent],
                )

    def visitation(self, env_state, traj_batch, final_obs):
        return None  # ipditm_stats(env_state, traj_batch, self.config.NUM_ENVS)

    @staticmethod
    def observation_space(env, env_params):
        return env.observation_space(env_params).shape

class Utils_KS(Utils_IMG):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def batchify_obs(x: dict, agent_list, num_agents, num_envs):
        # obs = jnp.stack([x[a]["observation"] for a in agent_list]).reshape(
        #     (num_agents, num_envs, *x[0]["observation"].shape[1:]))
        # inv = jnp.stack([x[a]["inventory"] for a in agent_list]).reshape((num_agents, num_envs, -1))
        # return (obs, inv)
        inter = jnp.stack([x[a] for a in agent_list])
        return inter.reshape((num_agents, num_envs, *inter.shape[2:]))

    @staticmethod
    def observation_space(env, env_params):
        return env.observation_space(env_params).shape

    def visitation(self, env_state, traj_batch, final_obs):
        return None
