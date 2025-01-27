import jax
from functools import partial
from gymnax.environments import spaces
import bsuite
import jax.numpy as jnp


class BsuiteToMARL(object):
    """ Use a Gymnax Environment within JaxMARL. Useful for debugging """

    num_agents = 1
    agent = 0
    agents = [agent]

    def __init__(self, env_name: str, env_kwargs: dict = {}):
        self.env_name = env_name
        # self._env, self.env_params = gymnax.make(env_name, **env_kwargs)
        self._env = bsuite.load_from_id(bsuite_id="deep_sea/1")

    # @property
    # def default_params(self):
    #     return self.env_params

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, params=None):
        # print('act', actions[self.agent])
        obs, state, reward, done, info = self._env.step(actions[self.agent])
        obs = (obs,)
        reward = (reward,)
        done = done  # {self.agent: done, "__all__": done}
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        timestep = self._env.reset()
        obs = state = timestep.observation
        obs = (obs,)
        return obs, state

    def observation_space(self, env_params):
        obs_spec = self._env.observation_spec()
        return spaces.Box(0, 1, (obs_spec.shape), jnp.float32)

    def action_space(self):
        action_spec = self._env.action_spec()
        return spaces.Discrete(action_spec.num_values)