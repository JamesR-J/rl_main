from absl import app
from project_name.baselines_run import run_train
import wandb
from project_name.config import get_config  # TODO dodge need to know how to fix this
import jax
from jax.lib import xla_bridge
import jax.profiler
import orbax
from flax.training import orbax_utils


jax.config.update("jax_enable_x64", True)  # TODO unsure if need or not but will check results


def main(_):
    config = get_config()

    # TODO can remove some memstate from those that do not need

    # TODO need to change update output to be for model info rather than env_state

    # TODO have changed PPO so double check it is correct to vanilla PPO and PPO_RNN and MFOS (as these are now correct)

    # TODO somehow add agent configs to wandb as well

    # TODO make rnn an option for all rather than diff agents, be cool if can still call agent PPO_RNN for example

    # actor = Agent(env=env, env_params=env_params, config=config, utils=None, key=key)
    #
    # config.AGENT_CONFIG =

    wandb.init(project="RL_BASE",
        entity=config.WANDB_ENTITY,
        config=config,
        # group="ks_tests",
        group="continuous_tests",
        mode=config.WANDB
    )

    config.DEVICE = jax.extend.backend.get_backend().platform
    print(config.DEVICE)

    with jax.disable_jit(disable=config.DISABLE_JIT):
        train = jax.jit(run_train(config))
        out = jax.block_until_ready(train())  # .block_until_ready()
        # jax.profiler.save_device_memory_profile("memory.prof")
        train_state = out["runner_state"][0][0]
        # critic_params = train_state.critic_state.params
        ckpt = {'model': train_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save('/home/james/PycharmProjects/rl_main/project_name/Orbax_Checkpoints/eyyooo', ckpt, save_args=save_args)


    print("FINITO")


if __name__ == '__main__':
    app.run(main)
