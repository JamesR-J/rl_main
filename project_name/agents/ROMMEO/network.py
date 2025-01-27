import sys

import flax.linen as nn
import functools

import jax.lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import distrax
from ml_collections import ConfigDict


class JointCriticROMMEO(nn.Module):  # TODO change this and remove RNN
    config: ConfigDict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, ego_action, opp_action):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # obs = nn.Dense(32)(obs)
        # ego_action = nn.Dense(16)(ego_action)
        # opp_action = nn.Dense(16)(opp_action)
        concat_obs = jnp.concatenate((obs, ego_action, opp_action), axis=-1)  # TODO pre with dense or not?

        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(concat_obs)
        critic = activation(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return jnp.squeeze(critic, axis=-1)


class ActorROMMEO(nn.Module):
    action_dim: Sequence[int]
    agent_config: ConfigDict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, opp_action = x

        concat_obs = jnp.concatenate((obs, opp_action), axis=-1)  # TODO pre with dense or not?

        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(concat_obs)
        embedding = activation(embedding)
        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)

        # if self.agent_config.DISCRETE:
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        return pi, jnp.zeros_like(actor_mean), jnp.zeros_like(actor_mean)

        # else:
        #     mu_and_logsig = nn.Dense(2 * self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        #
        #     mu = mu_and_logsig[..., :self.action_dim]
        #     log_sig = jnp.clip(mu_and_logsig[..., self.action_dim:], -20, 2)  # TODO hardcoded but maybe should change?
        #
        #     dist = distrax.MultivariateNormalDiag(mu, jnp.exp(log_sig) + 0.01)
        #
        #     return dist, mu, log_sig

class OppNetworkROMMEO(nn.Module):  # TODO think can combine the above
    action_dim: Sequence[int]
    agent_config: ConfigDict
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)

        # if self.agent_config.DISCRETE:
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        return pi, jnp.zeros_like(actor_mean), jnp.zeros_like(actor_mean)

        # else:
        #     mu_and_logsig = nn.Dense(2 * self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        #
        #     mu = mu_and_logsig[..., :self.action_dim]
        #     log_sig = jnp.clip(mu_and_logsig[..., self.action_dim:], -20, 2)  # TODO hardcoded but maybe should change?
        #
        #     dist = distrax.MultivariateNormalDiag(mu, jnp.exp(log_sig)+0.01)
        #
        #     return dist, mu, log_sig