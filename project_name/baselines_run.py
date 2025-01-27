import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.config import get_config  # TODO dodge need to know how to fix this
import wandb
import gymnax
from typing import NamedTuple
import chex
from .pax.envs.in_the_matrix import InTheMatrix, EnvParams as MatrixEnvParams
from .pax.envs.iterated_matrix_game import IteratedMatrixGame, EnvParams
from .pax.envs.coin_game import CoinGame
from .pax.envs.coin_game import EnvParams as CoinGameParams
from .agents import Agent, MultiAgent
from .utils import Transition, EvalTransition, Utils_IMG, Utils_IMPITM, Utils_CG, Utils_DEEPSEA, Utils_KS
import sys
from .gymnax_jaxmarl_wrapper import GymnaxToJaxMARL
from .deep_sea_wrapper import BsuiteToMARL
import bsuite
import jaxmarl
from .envs.KS_JAX import KS_JAX


"""
M - Number of Meta Episodes
E - Number of Episodes
L - Episode Length
G - Number of Agents
N - Number of Envs
O - Observation Dim
A - Action Dim
"""


def run_train(config):
    if config.CNN:
        payoff = jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]])
        env = InTheMatrix(num_inner_steps=config.NUM_INNER_STEPS, num_outer_steps=config.NUM_META_STEPS,
                          fixed_coin_location=False)
        env_params = MatrixEnvParams(payoff_matrix=payoff, freeze_penalty=5)
        utils = Utils_IMPITM(config)

        env = GymnaxToJaxMARL("DeepSea-bsuite", {"size": config.NUM_INNER_STEPS,
                                                 "sample_action_map": False})
        # check have updated the gymnax deep sea to the github change
        env_params = env.default_params
        utils = Utils_DEEPSEA(config)

        # env = bsuite.load_from_id(bsuite_id="deep_sea/1")
        # env = BsuiteToMARL("deep_sea/1")

    else:
        payoff = [[3, 3], [1, 4], [4, 1], [2, 2]]  # [[-1, -1], [-3, 0], [0, -3], [-2, -2]]  # payoff matrix for the IPD
        env = IteratedMatrixGame(num_inner_steps=config.NUM_INNER_STEPS, num_outer_steps=config.NUM_META_STEPS)
        env_params = EnvParams(payoff_matrix=payoff)
        utils = Utils_IMG(config)
        # TODO the above game has issues with when it ends? causing loss spikes it seems

        env = CoinGame(num_inner_steps=config.NUM_INNER_STEPS, num_outer_steps=config.NUM_META_STEPS,
                       cnn=False, egocentric=False)
        env_params = CoinGameParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])
        utils = Utils_CG(config)

        env = GymnaxToJaxMARL("KS_Equation", env=KS_JAX()) # TODO how to adjust default params for this step
        env_params = env.default_params
        utils = Utils_KS(config)

    # key = jax.random.PRNGKey(config.SEED)
    #
    # if config.NUM_AGENTS == 1:
    #     actor = Agent(env=env, env_params=env_params, config=config, utils=utils, key=key)
    # else:
    #     actor = MultiAgent(env=env, env_params=env_params, config=config, utils=utils, key=key)
    #
    # for agent in range(config.NUM_AGENTS):
    #     config[f"{actor.agent_types[agent]}_config"] = actor.agent_list[agent].agent_config()
    #
    # wandb.init(project="ProbInfMarl",
    #            entity=config.WANDB_ENTITY,
    #            config=config,
    #            group="coin-game_tests",
    #            mode=config.WANDB
    #            )
    # TODO sort out the above

    def train():
        key = jax.random.PRNGKey(config.SEED)

        if config.NUM_AGENTS == 1:
            actor = Agent(env=env, env_params=env_params, config=config, utils=utils, key=key)
        else:
            actor = MultiAgent(env=env, env_params=env_params, config=config, utils=utils, key=key)
        train_state, mem_state = actor.initialise()

        reset_key = jrandom.split(key, config.NUM_ENVS)
        obs_NO, env_state = jax.vmap(env.reset, in_axes=(0, None), axis_name="batch_axis")(reset_key, env_params)
        # TODO O may change above I guess

        runner_state = (
            train_state, mem_state, env_state, obs_NO, jnp.zeros((config.NUM_AGENTS, config.NUM_ENVS), dtype=bool), key)

        def _run_inner_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                train_state, mem_state, env_state, obs, last_done_GN, key = runner_state
                obs_batch_GNO = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)

                mem_state, action_GNA, log_prob_GN, value_GN, key = actor.act(train_state, mem_state, obs_batch_GNO, last_done_GN,
                                                                          key)

                # for not cnn
                # env_act = utils.unbatchify(action_n, range(config.NUM_AGENTS), config.NUM_AGENTS, config["NUM_DEVICES"])
                # env_act = {k: v for k, v in env_act.items()}
                # env_act = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), env_act)

                # for cnn maybe
                env_act_NGA = jnp.swapaxes(action_GNA, 0, 1)

                # step in env
                key, _key = jrandom.split(key)
                key_step = jrandom.split(_key, config.NUM_ENVS)
                obs, env_state, reward, done_N, info = jax.vmap(env.step, in_axes=(0, 0, 0, None),
                                                              axis_name="batch_axis")(key_step,
                                                                                      env_state,
                                                                                      env_act_NGA,
                                                                                      env_params
                                                                                      )
                info = jax.tree_map(lambda x: jnp.swapaxes(jnp.tile(x[:, jnp.newaxis], (1, config.NUM_AGENTS)), 0, 1),
                                    info)  # TODO not sure if need this basically
                done_batch_GN = jnp.swapaxes(jnp.tile(done_N[:, jnp.newaxis], (1, config.NUM_AGENTS)), 0, 1)
                reward_batch_GN = utils.batchify(reward, range(config.NUM_AGENTS), config.NUM_AGENTS,
                                              config["NUM_ENVS"]).squeeze(axis=-1)
                nobs_batch_GNO = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)

                mem_state = actor.update_encoding(train_state,
                                                  mem_state,
                                                  nobs_batch_GNO,
                                                  action_GNA,
                                                  reward_batch_GN,
                                                  done_batch_GN,
                                                  key)

                transition = Transition(done_batch_GN,
                                        done_batch_GN,  # TODO why are there two done batches?
                                        action_GNA,
                                        value_GN,
                                        reward_batch_GN,
                                        log_prob_GN,
                                        obs_batch_GNO,
                                        mem_state,
                                        # env_state,  # TODO have added for info purposes
                                        info,
                                        )

                return (train_state, mem_state, env_state, obs, done_batch_GN, key), transition

            runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config.NUM_INNER_STEPS)
            train_state, mem_state, env_state, obs, done, key = runner_state

            mem_state = actor.meta_act(mem_state)  # meta acts if using a meta agent, otherwise does nothing

            last_obs_batch = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)
            train_state, mem_state, env_state, last_obs_batch, done, agent_info, key = actor.update(train_state,
                                                                                                    mem_state,
                                                                                                    env_state,
                                                                                                    last_obs_batch,
                                                                                                    done,
                                                                                                    key,
                                                                                                    trajectory_batch)

            # update_steps = update_steps + 1

            return ((train_state, mem_state, env_state, obs, done, key), update_steps), (trajectory_batch, agent_info)

        def _run_meta_update(meta_runner_state, unused):
            (train_state, mem_state, env_state, obs, last_done, key), update_steps = meta_runner_state

            # reset env here actually I think
            # TODO this feels dodgy re ending of episodes in trajectories etc but seems what they have done
            key, _key = jrandom.split(key)
            reset_key = jrandom.split(_key, config.NUM_ENVS)
            obs, env_state = jax.vmap(env.reset, in_axes=(0, None), axis_name="batch_axis")(reset_key, env_params)

            # reset agents memory apparently as well, do I need this?
            mem_state = actor.reset_memory(mem_state)

            runner_state = (train_state, mem_state, env_state, obs,
                            jnp.zeros((config.NUM_AGENTS, config.NUM_ENVS), dtype=bool), key)

            update_state, (meta_trajectory_batch, agent_info) = jax.lax.scan(_run_inner_update,
                                                               (runner_state, update_steps),
                                                               None,
                                                               config.NUM_META_STEPS)
            collapsed_trajectory_batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x,
                                      [config.NUM_META_STEPS * config.NUM_INNER_STEPS, ] + list(x.shape[2:])),
                meta_trajectory_batch)

            runner_state, update_steps = update_state
            train_state, mem_state, env_state, obs, done, key = runner_state

            last_obs_batch = utils.batchify_obs(obs, range(config.NUM_AGENTS), config.NUM_AGENTS, config.NUM_ENVS)
            train_state, mem_state, env_state, last_obs_batch, done, meta_agent_info, key = actor.meta_update(train_state,
                                                                                                         mem_state,
                                                                                                         env_state,
                                                                                                         last_obs_batch,
                                                                                                         done, key,
                                                                                                         collapsed_trajectory_batch)

            def callback(traj_batch, env_stats, agent_info, meta_agent_info, update_steps):
                metric_dict = {  # "env_step": update_steps * config.NUM_ENVS * config.NUM_INNER_STEPS,
                                "env_stats": env_stats
                }

                # TODO below must be sooo slow but maybe it works fine?
                for step_idx in range(config.NUM_META_STEPS):
                    step_metric_dict = {}
                    for idx, agent in enumerate(config.AGENT_TYPE):
                        # shape is [num_meta_steps, num_inner_steps, num_agents, num_envs]
                        step_metric_dict[f"avg_reward_{agent}"] = traj_batch.reward[step_idx, :, idx, :].mean()
                        # step_metric_dict[f"avg_reward_{agent}_{idx}"] = traj_batch.reward[step_idx, :, idx, :].mean()
                        # step_metric_dict[f"avg_reward_{agent}_{idx}"] = traj_batch.reward[step_idx, -1, idx, :].mean()
                        # TODO have added the -1 for deepsea
                        if agent != "MFOS":
                            for item in agent_info[idx]:
                                step_metric_dict[f"{agent}-{item}"] = agent_info[idx][item][step_idx]
                                # step_metric_dict[f"{agent}_{idx}-{item}"] = agent_info[idx][item][step_idx]
                    wandb.log(step_metric_dict)

                for idx, agent in enumerate(config.AGENT_TYPE):
                    if agent == "MFOS":  # TODO update if get more meta agents
                        for item in meta_agent_info[idx]:
                            metric_dict[f"{agent}-{item}"] = meta_agent_info[idx][item]
                            # metric_dict[f"{agent}_{idx}-{item}"] = meta_agent_info[idx][item]
                # TODO have removed the idx from wandb

                metric_dict["env_step"] = (update_steps + 1) * config.NUM_META_STEPS

                wandb.log(metric_dict)

            env_stats = jax.tree_util.tree_map(lambda x: x.mean(), utils.visitation(env_state,
                                                                                    collapsed_trajectory_batch,
                                                                                    obs))
            # env_stats = env_state
            # TODO remove these stats if not using them and replace with deepsea stats somehow

            jax.experimental.io_callback(callback, None, meta_trajectory_batch,
                                         env_stats, agent_info, meta_agent_info, update_steps)

            metric = collapsed_trajectory_batch.info

            update_steps += 1

            return ((train_state, mem_state, env_state, obs, done, key), update_steps), metric

        # meta training is same between, just set =1 for no meta loops, I think?
        runner_state, metric = jax.lax.scan(_run_meta_update, (runner_state, 0), None, config.NUM_UPDATES)

        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
