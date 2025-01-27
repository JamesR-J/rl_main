import jax
import jax.numpy as jnp
import jax.random as jrandom

from functools import partial


class MDPBase:
    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def step(self):
        pass


class MDPPlus(MDPBase):
    def __init__(self):
        super().__init__()

    @partial(jax.jit, static_argnums=(0,))
    def reward(self):
        pass


class MDPMinus(MDPBase):
    def __init__(self):
        super().__init__()

    @partial(jax.jit, static_argnums=(0,))
    def reward(self):
        pass

