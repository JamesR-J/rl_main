import distrax
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv,
                                                     )
import jax.numpy as jnp
import jax
import jax.random as jrandom
import flax
from flax.training.train_state import TrainState
import random
import numpy as np
import optax
from project_name.vapor_stuff.config import get_config  # TODO dodge need to know how to fix this
from project_name.vapor_stuff.algos import SoftQNetwork, Actor, RandomisedPrior
import wandb
from project_name.vapor_stuff.buffer.prioritised_buffer import PrioritizedReplayBuffer


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


class TrainStateCritic(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    config = get_config()

    wandb.init(project="ProbInfMarl",
               entity=config.WANDB_ENTITY,
               config=config,
               group="TESTS",
               mode=config.WANDB
               )

    random.seed(config.SEED)  # TODO remove random
    np.random.seed(config.SEED)  # TODO remove numpy random
    key = jax.random.PRNGKey(config.SEED)
    key, actor_key, critic_key = jrandom.split(key, 3)

    envs = gym.vector.SyncVectorEnv(
        [make_env("BreakoutNoFrameskip-v4", config.SEED, i, False, "who knows m8") for i in range(config.NUM_ENVS)])

    # gym.register_envs(ale_py)

    obs, _ = envs.reset(seed=config.SEED)

    actor = Actor(action_dim=envs.single_action_space.n)
    critic = SoftQNetwork(action_dim=envs.single_action_space.n)
    randomised_prior = RandomisedPrior()

    actor_params = actor.init(actor_key, np.array([envs.single_observation_space.sample()]))
    critic_params = critic.init(critic_key, np.array([envs.single_observation_space.sample()]))

    actor_state = TrainState.create(apply_fn=None,
                                    params=actor_params,
                                    tx=optax.chain(optax.inject_hyperparams(optax.adam)(config.LR, eps=1e-4)),
                                    )
    critic_state = TrainStateCritic.create(apply_fn=None,
                                           params=critic_params,
                                           target_params=critic_params,
                                           tx=optax.chain(optax.inject_hyperparams(optax.adam)(config.LR, eps=1e-4)),
                                           )

    def create_reward_state(key, model, config):
        key, _key = jrandom.split(key)
        rp_params = model.init(_key, (np.array([envs.single_observation_space.sample()]), np.array([envs.single_action_space.sample()])[jnp.newaxis]))["params"]
        prior_params, reward_params = rp_params["static_prior"], rp_params["trainable"]
        reward_state = TrainState.create(apply_fn=randomised_prior.apply,
                                         params=reward_params,
                                         tx=optax.adam(config.LR))
        return prior_params, reward_state

    ensemble_keys = jrandom.split(key, config.NUM_ENSEMBLE)
    ensembled_prior_params, ensembled_reward_state = jax.vmap(create_reward_state, in_axes=(0, None, None))(ensemble_keys, randomised_prior, config)

    # Automatic entropy tuning
    if config.AUTOTUNE:
        target_entropy = -config.TARGET_ENT_SCALE * jnp.log(1 / jnp.array(envs.single_action_space.n))
        log_alpha = jnp.zeros(1)
        alpha = log_alpha.exp().item()
        a_optimizer = optax.Adam([log_alpha], lr=config.LR, eps=1e-4)
    else:
        alpha = config.ALPHA

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    rb = PrioritizedReplayBuffer(config.BUFFER_SIZE,
                                 envs.single_observation_space,
                                 envs.single_action_space,
                                 config.GAMMA,
                                 nstep=config.NSTEP,
                                 alpha=1.0,
                                 beta_steps=(config.TOTAL_TIMESTEPS - config.LEARNING_STARTS) / 1)

    # rb = ReplayBuffer(config.BUFFER_SIZE,
    #                   envs.single_observation_space,
    #                   envs.single_action_space,
    #                   "cpu",
    #                   optimize_memory_usage=True,
    #                   handle_timeout_termination=False,
    #                   )

    def get_action(actor_params, next_obs: jnp.ndarray, key: jrandom.PRNGKey):  # TODO double check this
        key, _key = jrandom.split(key)
        logits = actor.apply(actor_params, next_obs)
        policy_dist = distrax.Categorical(logits=logits)
        action = policy_dist.sample(seed=_key)
        log_prob = policy_dist.log_prob(action)
        action_probs = policy_dist.probs

        return action, log_prob[:, jnp.newaxis], action_probs

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=config.SEED)
    for global_step in range(config.TOTAL_TIMESTEPS):
        # ALGO LOGIC: put action logic here
        # epsilon = linear_schedule(config.START_EPS, config.END_EPS, config.EPS_DECAY * config.TOTAL_TIMESTEPS,
        #                           global_step)
        if global_step < config.LEARNING_STARTS:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # print(obs.shape)
            actions, _, _ = get_action(actor_state.params, obs, key)

        # print(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    wandb.log({"episodic_return": info["episode"]["r"],
                               "episodic_length": info["episode"]["l"]})

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        def get_mask(truncations, done):
            return done if not truncations else False

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        # rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        mask = get_mask(truncations, terminations)
        rb.append(obs, actions, rewards, mask, next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs  # issue with this is means obs gets overruled in the below functions :((

        # ALGO LOGIC: training.
        if global_step > config.LEARNING_STARTS:
            if global_step % config.TRAIN_FREQ == 0:
                # data = rb.sample(config.BATCH_SIZE)
                weight, batch = rb.sample(config.BATCH_SIZE)
                state, action, reward, done, nstate = batch

                _, next_state_log_pi, next_state_action_probs = get_action(actor_params, nstate, key)
                qf1_next_target = critic.apply(critic_state.target_params, nstate)

                # CRITIC training
                def critic_loss(critic_params, obs, actions, rewards, dones, next_state_log_pi, next_state_action_probs, qf1_next_target, key):
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (qf1_next_target - alpha * next_state_log_pi)
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(axis=1)[:, jnp.newaxis]
                    next_q_value = rewards + (1 - dones) * config.GAMMA * (min_qf_next_target)

                    # use Q-values only for the taken actions
                    qf1_values = critic.apply(critic_params, obs)
                    qf1_a_values = jnp.take_along_axis(qf1_values, actions, axis=1)

                    abs_td = jnp.abs(qf1_a_values - next_q_value)

                    # mse loss below
                    qf_loss = jnp.mean((qf1_a_values - next_q_value) ** 2)  # TODO check this is okay?

                    return qf_loss, (jax.lax.stop_gradient(qf1_a_values), jax.lax.stop_gradient(abs_td))

                (critic_loss, (q_pred, abs_td)), grads = jax.value_and_grad(critic_loss, has_aux=True)(
                    critic_state.params,
                    state,
                    action,
                    reward,
                    done,
                    next_state_log_pi,
                    next_state_action_probs,
                    qf1_next_target,
                    key)

                rb.update_priority(abs_td)

                critic_state = critic_state.apply_gradients(grads=grads)

                min_qf_values = critic.apply(critic_state.params, state)

                def actor_loss(actor_params, obs, min_qf_values, key):
                    # ACTOR training
                    _, log_pi, action_probs = get_action(actor_params, obs, key)

                    # no need for reparameterisation, the expectation can be calculated for discrete actions
                    actor_loss = jnp.mean(action_probs * ((alpha * log_pi) - min_qf_values))

                    return actor_loss, jax.lax.stop_gradient(action_probs)

                (loss_value, action_probs2), grads = jax.value_and_grad(actor_loss, has_aux=True)(
                    actor_state.params,
                    state,
                    min_qf_values,
                    key
                )  # TODO do we need action_probs2? nah lol just added cus cba to do has_aux
                actor_state = actor_state.apply_gradients(grads=grads)

                def train_ensemble(train_state, prior_params, obs, actions, rewards):
                    def reward_predictor_loss(rp_params):
                        rew_pred = randomised_prior.apply({"params": {"static_prior": prior_params, "trainable": rp_params}}, (obs, actions))
                        loss = jnp.mean((rew_pred - rewards) ** 2)
                        return loss, rew_pred

                    (loss, reward_preds), grads = jax.value_and_grad(reward_predictor_loss, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state

                ensemble_state = jnp.repeat(state[jnp.newaxis, :], config.NUM_ENSEMBLE, axis=0)
                ensemble_action = jnp.repeat(action[jnp.newaxis, :], config.NUM_ENSEMBLE, axis=0)

                key, _key = jrandom.split(key)
                jitter_reward = reward + config.RP_NOISE * jrandom.normal(_key, shape=reward.shape)  # TODO should this really be done once to each data point?
                ensemble_reward = jnp.repeat(jitter_reward[jnp.newaxis, :], config.NUM_ENSEMBLE, axis=0)

                ensembled_reward_state = jax.vmap(train_ensemble)(ensembled_reward_state, ensembled_prior_params,
                                                                  ensemble_state,
                                                                  ensemble_action,
                                                                  ensemble_reward)

                # if args.autotune:  # TODO add autotune entropy feature at some point
                #     # re-use action probabilities for temperature loss
                #     alpha_loss = (
                #                 action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()
                #
                #     a_optimizer.zero_grad()
                #     alpha_loss.backward()
                #     a_optimizer.step()
                #     alpha = log_alpha.exp().item()
                # # to here so need to replace all

            # update target network
            if global_step % config.TARGET_NETWORK_FREQ == 0:
                critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params,
                                                                                           critic_state.target_params,
                                                                                           config.TAU)
                                                    )

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     with open(model_path, "wb") as f:
    #         f.write(flax.serialization.to_bytes(q_state.params))
    #     print(f"model saved to {model_path}")
    #     from cleanrl_utils.evals.dqn_jax_eval import evaluate

    # episodic_returns = evaluate(
    #     model_path,
    #     make_env,
    #     args.env_id,
    #     eval_episodes=10,
    #     run_name=f"{run_name}-eval",
    #     Model=QNetwork,
    #     epsilon=0.05,
    # )
    # for idx, episodic_return in enumerate(episodic_returns):
    #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()


if __name__ == "__main__":
    main()
