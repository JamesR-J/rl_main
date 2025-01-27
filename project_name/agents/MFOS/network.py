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


class ScannedMFOSRNN(nn.Module):
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
        obs, inventory = obsinv
        conv_layer = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(obs)
        conv_layer = nn.relu(conv_layer)
        flatten_layer = conv_layer.reshape((obs.shape[0], obs.shape[1], -1))  # TODO check this
        concat_layer = jnp.concatenate((flatten_layer, inventory), axis=-1)
        last_layer = nn.Dense(16)(concat_layer)  # TODO maybe 16 as a hyperparam
        return last_layer


class ActorCriticMFOSRNN(nn.Module):
    action_dim: Sequence[int]
    config: ConfigDict
    agent_config: ConfigDict

    @nn.compact
    def __call__(self, hidden, x, th):
        obs, dones = x
        hidden_t, hidden_a, hidden_c = jnp.split(hidden, 3, axis=-1)  # TODO check this ting I guess

        # embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        # embedding = nn.relu(embedding)
        if self.config.CNN:
            meta_emb = CNNtoLinear()(obs)
            actor_emb = CNNtoLinear()(obs)
            critic_emb = CNNtoLinear()(obs)
        else:
            meta_emb = nn.Dense(self.agent_config.GRU_HIDDEN_DIM // 3, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            actor_emb = nn.Dense(self.agent_config.GRU_HIDDEN_DIM // 3, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            critic_emb = nn.Dense(self.agent_config.GRU_HIDDEN_DIM // 3, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        # TODO theres no non lineariites??

        hidden_a, embedding_a = ScannedMFOSRNN()(hidden_a, (actor_emb, dones))
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(th * embedding_a)
        pi = distrax.Categorical(logits=actor_mean)

        hidden_c, embedding_c = ScannedMFOSRNN()(hidden_c, (critic_emb, dones))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(embedding_c)

        hidden_t, embedding_t = ScannedMFOSRNN()(hidden_t, (meta_emb, dones))
        current_th = nn.sigmoid(nn.Dense(self.agent_config.GRU_HIDDEN_DIM // 3)(embedding_t))

        hidden = jnp.concatenate([hidden_t, hidden_a, hidden_c], axis=-1)

        # print(current_th)

        return hidden, pi, jnp.squeeze(critic, axis=-1), actor_mean, current_th
