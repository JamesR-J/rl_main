import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.config import get_config  # TODO dodge need to know how to fix this
import wandb
import gymnax
from typing import NamedTuple
import chex
from .agents import Agent
from .utils import Transition, EvalTransition, Utils_IMG, Utils_DEEPSEA, Utils_KS, Utils_Cartpole
import sys
from .deep_sea_wrapper import BsuiteToMARL
import bsuite
from .envs.KS_JAX import KS_JAX
from .envs.wrappers import LogWrapper, FlattenObservationWrapper


def run_train(config):
    if config.DISCRETE:
        env, env_params = gymnax.make("CartPole-v1")
        env = LogWrapper(env)

        # env, env_params = gymnax.make("Acrobot-v1")
        # env = LogWrapper(env)
        #
        env, env_params = gymnax.make("DeepSea-bsuite", size=config.DEEP_SEA_MAP, sample_action_map=True)
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)

        # env, env_params = gymnax.make("MountainCar-v0")
        # env = LogWrapper(env)
    else:
        env = KS_JAX() # TODO how to adjust default params for this step
        env_params = env.default_params
        env = LogWrapper(env)  # TODO does this work with the env?

        env, env_params = gymnax.make("MountainCarContinuous-v0")
        env = LogWrapper(env)
        #
        # env, env_params = gymnax.make("Swimmer-misc")
        # env = LogWrapper(env)

    # key = jax.random.PRNGKey(config.SEED)
    #
    # actor = Agent(env=env, env_params=env_params, config=config, utils=utils, key=key))
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

        actor = Agent(env=env, env_params=env_params, config=config, utils=None, key=key)
        train_state, mem_state = actor.initialise()

        reset_key = jrandom.split(key, config.NUM_ENVS)
        init_obs_NO, env_state = jax.vmap(env.reset, in_axes=(0, None), axis_name="batch_axis")(reset_key, env_params)

        runner_state = (train_state, mem_state, env_state, init_obs_NO, jnp.zeros(config.NUM_ENVS, dtype=bool), key)

        def _run_inner_update(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                train_state, mem_state, env_state, obs_NO, done_N, key = runner_state

                mem_state, action_NA, key = actor.act(train_state, mem_state, obs_NO, done_N, key)

                # step in env
                # key, _key = jrandom.split(key)
                key_step = jrandom.split(key, config.NUM_ENVS)
                nobs_NO, env_state, reward_N, ndone_N, info = jax.vmap(env.step, in_axes=(0, 0, 0, None),
                                                              axis_name="batch_axis")(key_step,
                                                                                      env_state,
                                                                                      action_NA,
                                                                                      env_params
                                                                                      )

                # mem_state = actor.update_encoding(train_state, mem_state, nobs_NO, action_NA, reward_N, ndone_N, key)

                transition = Transition(ndone_N, action_NA, reward_N, obs_NO, mem_state,
                                        # env_state,  # TODO have added for info purposes
                                        info)

                return (train_state, mem_state, env_state, nobs_NO, ndone_N, key), transition

            ((train_state, mem_state, env_state, last_obs_NO, last_done_N, key),
             trajectory_batch_LNZ) = jax.lax.scan(_run_episode_step, runner_state, None, config.NUM_INNER_STEPS)

            train_state, mem_state, agent_info, key = actor.update(train_state, mem_state,  last_obs_NO,  last_done_N,
                                                                   key, trajectory_batch_LNZ)

            def callback(traj_batch, env_stats, agent_stats, update_steps):
                metric_dict = {"Total Steps": update_steps * config.NUM_ENVS * config.NUM_INNER_STEPS,
                               "Total_Episodes": update_steps * config.NUM_ENVS,
                               "avg_reward": traj_batch.reward.mean(),
                               "avg_returns": traj_batch.info["returned_episode_returns"][traj_batch.info["returned_episode"]].mean(),
                               "avg_episode_end_reward": traj_batch.info["reward"][traj_batch.info["returned_episode"]].mean(),
                               "avg_action": traj_batch.action.mean()}

                # shape is LN, so we are averaging over the num_envs and episode
                for item in agent_info:
                    metric_dict[f"{item}"] = agent_stats[item]

                print(traj_batch.reward)

                # print(traj_batch.info["reward"][traj_batch.info["returned_episode"]].mean())

                wandb.log(metric_dict)

            # env_stats = jax.tree_util.tree_map(lambda x: x.mean(), utils.visitation(env_state,
            #                                                                         collapsed_trajectory_batch,
            #                                                                         obs))

            jax.experimental.io_callback(callback, None, trajectory_batch_LNZ, env_state,
                                         agent_info, update_steps)

            update_steps = update_steps + 1

            return ((train_state, mem_state, env_state, last_obs_NO, last_done_N, key), update_steps), None

        runner_state, _ = jax.lax.scan(_run_inner_update,(runner_state, 0),None, config.NUM_EPISODES)

        return {"runner_state": runner_state}

    return train


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
