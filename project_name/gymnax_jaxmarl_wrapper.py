"""
Based off: https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/wrappers/gymnax.py
"""

import jax
from functools import partial
import gymnax


class GymnaxToJaxMARL(object):
    """ Use a Gymnax Environment within JaxMARL. Useful for debugging """

    num_agents = 1
    agent = 0
    agents = [agent]

    def __init__(self, env_name: str, env_kwargs: dict = {}, env=None):
        self.env_name = env_name

        if env is None:
            self._env, self.env_params = gymnax.make(env_name, **env_kwargs)
        else:
            self._env = env
            self.env_params = env.params

    @property
    def default_params(self):
        return self.env_params

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, params=None):
        # print('act', actions[self.agent])
        obs, state, reward, done, info = self._env.step(key, state, actions[self.agent], params)
        obs = (obs,)
        reward = (reward,)
        done = done  # {self.agent: done, "__all__": done}
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        obs = (obs,)
        return obs, state

    def observation_space(self, env_params):
        return self._env.observation_space(env_params)

    def action_space(self):
        return self._env.action_space(self.env_params)