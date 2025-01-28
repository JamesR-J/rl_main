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


class DiscreteActorCritic(nn.Module):
    action_dim: int
    config: ConfigDict
    agent_config: ConfigDict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.config.CNN:
            embedding = CNNtoLinear()(obs)
        else:
            embedding = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            embedding = nn.relu(embedding)

        embedding = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        embedding = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1), actor_mean


class ContinuousActorCritic(nn.Module):
    action_dim: int
    config: ConfigDict
    agent_config: ConfigDict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.config.CNN:
            embedding = CNNtoLinear()(obs)
        else:
            embedding = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            embedding = nn.relu(embedding)

        embedding = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        embedding = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        actor_logstd = nn.tanh(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding))
        actor_logstd = self.agent_config.LOGSTD_MIN + 0.5 * (self.agent_config.LOGSTD_MAX - self.agent_config.LOGSTD_MIN) * (actor_logstd + 1)

        pi = distrax.Normal(actor_mean, jnp.exp(actor_logstd))

        critic = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1), actor_mean


class SimpleNetwork(nn.Module):
    action_dim: int
    config: ConfigDict
    agent_config: ConfigDict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, actions):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.config.CNN:
            obs = CNNtoLinear()(obs)

        obs = nn.Dense(self.agent_config.HIDDEN_SIZE - self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        x = jnp.concatenate((obs, actions), axis=-1)

        x = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.agent_config.HIDDEN_SIZE, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

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
    def __call__(self, obs, actions):
        return self._net(obs, actions) + self.agent_config.PRIOR_SCALE * self._prior_net(obs, actions)


