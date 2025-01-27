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
    def __call__(self, obs):
        flatten_layer = jnp.reshape(obs, (obs.shape[0], obs.shape[1], -1))
        return flatten_layer


class ContinuousRNNQNetwork(nn.Module):  # TODO change this and remove RNN
    config: ConfigDict
    activation: str = "tanh"
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, x):
        (obs, actions), dones = x

        # if self.config.CNN:  # TODO turned off CNN as well
        #     embedding = CNNtoLinear()(obs)
        # else:
        #     embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        #     embedding = nn.relu(embedding)

        # rnn_in = (embedding, dones)  # TODO turned off rnn for now
        # hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hidden = None

        s1 = nn.swish(nn.Dense(256)(obs))
        s2 = nn.swish(nn.Dense(128)(s1))
        a1 = nn.swish(nn.Dense(128)(actions))

        new_x = jnp.concatenate((s2, a1), axis=-1)
        new_x = nn.swish(nn.Dense(128)(new_x))
        q_vals = nn.Dense(1)(new_x)

        return hidden, q_vals


class DeterministicPolicy(nn.Module):
    action_dim: int
    config: ConfigDict
    action_scale: float
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        obs, dones = x

        # if self.config.CNN:  # TODO turned off CNN as well
        #     embedding = CNNtoLinear()(obs)
        # else:
        #     embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        #     embedding = nn.relu(embedding)

        # rnn_in = (embedding, dones)  # TODO turned off rnn for now
        # hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hidden = None

        new_x = nn.relu(nn.Dense(256)(obs))  # TODO sort out INITs
        new_x = nn.relu(nn.Dense(128)(new_x))
        new_x = nn.relu(nn.Dense(64)(new_x))

        action = nn.tanh(nn.Dense(self.action_dim)(new_x))

        action = action * self.action_scale

        return hidden, action