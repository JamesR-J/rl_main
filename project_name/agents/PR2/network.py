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


class JointCriticPR2(nn.Module):  # TODO change this and remove RNN
    config: ConfigDict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, ego_action, opp_action):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, dones = x  # TODO some how obs also has actions and opponent actions included

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


class IndCriticPR2(nn.Module):  # TODO change this and remove RNN
    config: Dict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, ego_action):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, dones = x  # TODO some how obs also has actions and opponent actions included

        # obs = nn.Dense(32)(obs)
        # ego_action = nn.Dense(16)(ego_action)
        # opp_action = nn.Dense(16)(opp_action)
        concat_obs = jnp.concatenate((obs, ego_action), axis=-1)  # TODO pre with dense or not?

        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(concat_obs)
        critic = activation(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return jnp.squeeze(critic, axis=-1)


class ActorPR2(nn.Module):  # TODO change this and remove RNN
    action_dim: Sequence[int]
    config: Dict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, dones = x  # TODO some how obs also has actions and opponent actions included

        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)
        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)

        pi = distrax.Categorical(logits=actor_mean)

        return pi, actor_mean

class OppNetworkPR2(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, actions, latents):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        concat_obs = jnp.concatenate((obs, actions, latents), axis=-1)  #
        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(concat_obs)
        embedding = activation(embedding)
        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        actions = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)

        return actions  # TODO probs needs some scaling and sorting eg discrete or continuous etc