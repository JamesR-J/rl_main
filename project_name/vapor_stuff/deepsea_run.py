import jax.numpy as jnp
import jax
import jax.random as jrandom
from project_name.vapor_stuff.config import get_config  # TODO dodge need to know how to fix this
import wandb
import gymnax
from project_name.vapor_stuff.algos import VAPOR_Lite, SAC, VAPOR_Lite_Less_Discrete
from typing import NamedTuple
import chex
from project_name.vapor_stuff.utils import TransitionNoInfo
from project_name.vapor_stuff.envs.adjusted_gymnax_deep_sea import DeepSea


class Transition(NamedTuple):
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    ensemble_reward: chex.Array
    done: chex.Array
    logits: chex.Array
    info: jnp.ndarray


def run_train(config):
    # env, env_params = gymnax.make("DeepSea-bsuite", size=config.DEEPSEA_SIZE)  # TODO edited the gymnax env for deepsea for info
    env = DeepSea(size=config.DEEPSEA_SIZE)
    env_params = env.default_params

    def train():
        key = jax.random.PRNGKey(config.SEED)

        actor = VAPOR_Lite(env, env_params, key, config)
        # actor = VAPOR_Lite_Less_Discrete(env, env_params, key, config)
        # actor = SAC(env, env_params, key, config)

        # ensrpr relates to ensembled randomised prior reward state
        actor_state, critic_state, ensrpr_state, buffer_state, key = actor.create_train_state(key)

        key, _key = jrandom.split(key)
        obs, env_state = env.reset(_key, env_params)

        runner_state = (
            actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, jnp.zeros((), dtype=bool), key)

        def _run_update(update_runner_state, unused):
            runner_state, update_steps, episodes_finished = update_runner_state

            def _run_episode_step(runner_state, unused):
                # take initial env_state
                actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, last_done, key = runner_state

                def random_act(actor_params, obs, key):
                    key, _key = jrandom.split(key)
                    action = env.action_space(env_params).sample(rng=_key)
                    return action[jnp.newaxis], jnp.zeros((1, 1)), jnp.zeros((1, 2)), key

                # act on this initial env_state if above certain outer steps
                # action, _, _, key = jax.lax.cond(update_steps > config.LEARNING_STARTS,
                #                                                    actor.act,
                #                                                    random_act,
                #                                                    actor_state.params,
                #                                                    obs[jnp.newaxis, :, :, jnp.newaxis],
                #                                                    key
                #                                                    )
                action, _, _, logits, key = actor.act(actor_state.params, obs[jnp.newaxis, :, :, jnp.newaxis], key)
                # action = jnp.ones((1), dtype=int)

                # step in env
                key, _key = jrandom.split(key)
                nobs, nenv_state, reward, done, info = env.step(_key,
                                                                env_state,
                                                                action[0],
                                                                env_params)

                key, _key = jrandom.split(key)
                jitter_reward = reward[jnp.newaxis] + config.RP_NOISE * jrandom.normal(_key, shape=reward[jnp.newaxis].shape)

                info["bad_episode"] = env_state.env_state.bad_episode

                # update tings my dude
                transition = Transition(obs[:, :, jnp.newaxis], action, reward[jnp.newaxis], jitter_reward, done[jnp.newaxis], jnp.squeeze(logits, axis=0), info)

                return (actor_state, critic_state, ensrpr_state, buffer_state, nenv_state, nobs, done, key), transition

            # TODO maybe run for data collection for a while until learning starts

            # run for NUM_STEPS length rollout
            runner_state, trajectory_batch = jax.lax.scan(_run_episode_step, runner_state, None, config["NUM_STEPS"])
            actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key = runner_state

            buffer_state = actor.per_buffer.add(buffer_state, TransitionNoInfo(state=trajectory_batch.state,
                                                                               action=trajectory_batch.action,
                                                                               reward=trajectory_batch.reward,
                                                                               ensemble_reward=trajectory_batch.ensemble_reward,
                                                                               done=trajectory_batch.done,
                                                                               logits=trajectory_batch.logits))
            # TODO maybe refresh the above so a bit better
            # needs the below to add the new trajectory_buffer
            runner_state = actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key

            # update agents here after rollout
            # def fake_update_fn(runner_state):
            #     actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key = runner_state
            #     return actor_state, critic_state, ensrpr_state, buffer_state, key
            # actor_state, critic_state, ensrpr_state, buffer_state, key = jax.lax.cond(actor.per_buffer.can_sample(buffer_state),
            #                                                                           actor.update,
            #                                                                           fake_update_fn,
            #                                                                           runner_state)

            actor_state, critic_state, ensrpr_state, buffer_state, actor_loss, critic_loss, mean_ensembled_loss, pg_loss, ent_loss, key = actor.update(
                runner_state)

            # metric handling
            # metric = jax.tree_map(lambda x: jnp.swapaxes(x, 1, 2), trajectory_batch.info)
            metric = trajectory_batch.info

            def callback(metric, actor_loss, critic_loss, mean_ensembled_loss, pg_loss, ent_loss):
                # print(metric["update_steps"])
                # print(train_state.params)#["Dense_2"])
                # if metric["update_steps"] == 2:
                #     sys.exit()
                metric_dict = {
                    "returns": metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                    "episode_won_total": jnp.sum(~metric["bad_episode"][metric["returned_episode"]]),
                    "episodes_finished": jnp.sum(metric["returned_episode"]),
                    "episode_won_average": jnp.sum(~metric["bad_episode"][metric["returned_episode"]]) / metric["returned_episode"].sum(),
                    "env_step": metric["update_steps"] * config.NUM_ENVS * config.NUM_STEPS,
                    "episode_finished": metric["episode_finished"],
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "pg_loss": pg_loss,
                    "ent_loss": ent_loss,
                    "mean_ensembled_loss": mean_ensembled_loss
                }
                wandb.log(metric_dict)

            metric["update_steps"] = update_steps
            episodes_finished += jnp.sum(metric["returned_episode"])
            metric["episode_finished"] = episodes_finished
            jax.experimental.io_callback(callback, None,
                                         metric,
                                         actor_loss,
                                         critic_loss,
                                         mean_ensembled_loss,
                                         pg_loss,
                                         ent_loss)

            update_steps = update_steps + 1

            return (
            (actor_state, critic_state, ensrpr_state, buffer_state, env_state, obs, done, key), update_steps, episodes_finished), metric

        runner_state, metric = jax.lax.scan(_run_update, (runner_state, 0, 0), None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = get_config()
    with jax.disable_jit(disable=True):
        train = run_train(config)
        out = train()
