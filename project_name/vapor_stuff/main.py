from absl import app, flags
from project_name.vapor_stuff.deepsea_run import run_train
import wandb
from project_name.vapor_stuff.config import get_config  # TODO dodge need to know how to fix this
import jax
from jax.lib import xla_bridge
import sys


_SEED = flags.DEFINE_integer("seed", 44, "Random seed")
_DEEPSEA_DEPTH = flags.DEFINE_integer("deepsea_depth", 12, "Depth of deepsea environment")
_DISABLE_JIT = flags.DEFINE_boolean("disable_jit", False, "To disable jit")


def main(_):
    config = get_config()

    # TODO needs to add the v-trace at some point but kinda cba for now
    # TODO should be fairly easy using rlax if vmap the loss?

    config.SEED = _SEED.value
    config.DEEPSEA_SIZE = _DEEPSEA_DEPTH.value
    config.DISABLE_JIT = _DISABLE_JIT.value
    config.NUM_STEPS = config.DEEPSEA_SIZE

    config.DEVICE = xla_bridge.get_backend().platform

    wandb.init(  # project="ProbInfMarl",
               #  entity=config.WANDB_ENTITY,
               config=config,
               # group="DEEPSEA_ISH",
               # mode=config.WANDB
               )

    with jax.disable_jit(disable=config.DISABLE_JIT):
        train = jax.jit(run_train(config))
        out = jax.block_until_ready(train())


if __name__ == '__main__':
    app.run(main)
