import jax
import jax.numpy as jnp
import jax.random as jrandom
from ..utils import import_class_from_folder  # , batchify
from functools import partial
from typing import Any, Dict, Tuple
import chex
import sys
from flax.training.train_state import TrainState


# initialise agents from the config file deciding what the algorithms are
class Agent:
    def __init__(self, env, env_params, config, utils, key: chex.PRNGKey):  # TODO add better chex
        self.env = env
        self.config = config
        self.utils = utils
        self.agent_type = config.AGENT_TYPE
        self.agent = import_class_from_folder(self.agent_type)(env=env, env_params=env_params, key=key, config=config, utils=utils)
        self.train_state, self.mem_state = self.agent.create_train_state()

    @partial(jax.jit, static_argnums=(0,))
    def initialise(self):
        return self.train_state, self.mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: chex.Array, obs_batch: Dict[str, chex.Array], last_done: chex.Array,
            key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.PRNGKey]:  # TODO add better chex for trainstate
        ac_in = (obs_batch,
                 last_done,
                 )
        mem_state, action, key = self.agent.act(train_state, mem_state, ac_in, key)
        return mem_state, action, key

    @partial(jax.jit, static_argnums=(0,))
    def update_encoding(self, train_state: Any, mem_state: Any, obs_batch: Any, action: Any, reward: Any, done: Any, key):
        agent = 0
        return self.agent.update_encoding(train_state, mem_state, agent, obs_batch, action, reward, done, key)

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state: Any):
        return self.agent.reset_memory(mem_state)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, train_state: Any, mem_state: Any, last_obs_batch: Any, last_done: Any, key: Any,
               trajectory_batch: Any):  # TODO add better chex
        ac_in = (last_obs_batch, last_done)
        train_state = (train_state, mem_state, ac_in, key)
        train_state, mem_state, info, key = self.agent.update(train_state, 0, trajectory_batch, None)

        return train_state, mem_state, info, key
