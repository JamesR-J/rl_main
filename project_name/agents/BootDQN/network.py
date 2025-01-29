import sys

import flax.linen as nn
import functools
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import distrax
from ml_collections import ConfigDict
import jax


class CNNtoLinear(nn.Module):
    @nn.compact
    def __call__(self, obs):
        flatten_layer = jnp.reshape(obs, (obs.shape[0], obs.shape[1], -1))
        return flatten_layer


class SimpleNetwork(nn.Module):
    action_dim: int
    config: ConfigDict
    agent_config: ConfigDict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):  # , actions):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.config.CNN:
            obs = CNNtoLinear()(obs)

        # obs = nn.Dense(self.agent_config.HIDDEN_SIZE - self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        # x = jnp.concatenate((obs, actions), axis=-1)
        x = obs

        x = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        return x


class EnsembleNetwork(nn.Module):
    action_dim: int
    config: ConfigDict
    agent_config: ConfigDict
    activation: str = "tanh"

    def setup(self):
        self._net = SimpleNetwork(self.action_dim, self.config, self.agent_config)
        self._prior_net = SimpleNetwork(self.action_dim, self.config, self.agent_config)

    @nn.compact
    def __call__(self, obs):
        return self._net(obs) + self.agent_config.PRIOR_SCALE * self._prior_net(obs)


