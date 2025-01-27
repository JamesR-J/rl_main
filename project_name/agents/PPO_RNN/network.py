import sys

import flax.linen as nn
import functools
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import distrax
import seaborn as sns
import matplotlib.pyplot as plt
from ml_collections import ConfigDict


class ScannedRNN(nn.Module):
    @functools.partial(nn.scan,
                       variable_broadcast="params",
                       in_axes=0,
                       out_axes=0,
                       split_rngs={"params": False},
                       )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        rnn_state = jnp.where(resets[:, jnp.newaxis],
                              self.initialize_carry(*rnn_state.shape),
                              rnn_state,
                              )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jrandom.PRNGKey(0), (batch_size, hidden_size))


class CNNtoLinear(nn.Module):
    @nn.compact
    def __call__(self, obsinv):
        # obs, inventory = obsinv  # TODO some check if more than one dimension to split but otherwise leave it
        obs = obsinv
        conv_layer = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(obs)
        conv_layer = nn.relu(conv_layer)
        flatten_layer = conv_layer.reshape((obs.shape[0], obs.shape[1], -1))  # TODO check this
        concat_layer = flatten_layer  # jnp.concatenate((flatten_layer, inventory), axis=-1)
        last_layer = nn.Dense(16)(concat_layer)  # TODO maybe 16 as a hyperparam
        return last_layer


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: ConfigDict
    agent_config: ConfigDict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.config.CNN:
            embedding = CNNtoLinear()(obs)
        else:
            embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.agent_config.GRU_HIDDEN_DIM, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1), actor_mean
