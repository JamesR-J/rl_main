import chex
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jrandom
import flax.linen as nn
from functools import partial
from typing import Sequence
import sys


class EncoderRNN(nn.Module):
    @partial(nn.scan,
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

        rnn_state = jnp.where(resets[:, np.newaxis],
                              self.initialize_carry(*rnn_state.shape),
                              rnn_state,
                              )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class DecoderRNN(nn.Module):
    @partial(nn.scan,
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
        rnn_state = jnp.where(resets[:, np.newaxis],
                              self.initialize_carry(*rnn_state.shape),
                              rnn_state,
                              )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class Encoder(nn.Module):
    """
    Need to differentiate between one-step predictions and entire trajectories? But this also changes size of decoder,
    need to figure this bit out

    For now assuming always have full trajectories that are fed in so the first step should have no hidden state
    """
    config: dict

    @nn.compact
    def __call__(self, hidden, past_traj, full_traj=False):
        # shape batch_size, num_envs, len_trajectory, embedding  # TODO for now need to double check dimensions though
        state, action, reward, dones = past_traj
        state = nn.relu(nn.Dense(32)(state))
        action = nn.relu(nn.Dense(16)(action))
        reward = nn.relu(nn.Dense(16)(reward))

        rnn_input = jnp.concatenate([state, action, reward], axis=-1)

        # add a recurrent layer of size 64
        rnn_input = (rnn_input, dones)  # adds middle num_envs extra dim for computation
        # rnn_input[0].shape should be traj_len, batch_size (aka num_envs), embedding_dim
        hidden, embedding = EncoderRNN()(hidden, rnn_input)

        # then another size 64 that is non recurrent
        post_rnn = nn.relu(nn.Dense(64)(embedding))

        # then another size 64xlatent_dim
        final_layer = nn.relu(nn.Dense(64)(post_rnn))  # TODO figure out how to get multiple agents here

        # output should be n (number of agent) dimensions of mu and sigma
        mu = nn.Dense(self.config.LATENT_DIM)(final_layer)
        log_sigma = nn.Dense(self.config.LATENT_DIM)(final_layer)

        return hidden, mu, log_sigma


class Decoder(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, hidden, dones, latent_space, state):
        # TODO hidden state generation from the first step of the decoder rnn, otherwise use previous hidden state
        state = nn.relu(nn.Dense(32)(state))
        initial_layer = jnp.concatenate([state, latent_space], axis=-1)

        # then layer 32xlatent_dim
        m_dim = nn.relu(nn.Dense(32)(initial_layer))

        # then layer 64xlatent_dim
        mt_dim = nn.relu(nn.Dense(64)(m_dim))

        # 64
        inter_layer = nn.relu(nn.Dense(64)(mt_dim))

        # 64 recurrent layer
        rnn_input = (inter_layer, dones)
        hidden, embedding = DecoderRNN()(hidden, rnn_input)  # TODO could use same rnn?
        # embedding = inter_layer

        # 32
        last_embedding = nn.relu(nn.Dense(32)(embedding))

        # opponent_action_logits = nn.Dense(2)(last_embedding)
        opponent_action_logits = nn.Dense(self.action_dim)(last_embedding)

        # shape should be batch_size, num_envs, action_dim
        return hidden, opponent_action_logits  # TODO dimensions should be number of forecasting steps idk what the dims be here tho


class HierarchicalSequentialVAE(nn.Module):
    """
    1) it can train on full trajectories and then it needs to reset prior when new episode begins?
    2) if one step prediction then can feed in old hidden state and not need the prior
    3) want to be able to call encoder only for the policy step
    4) should we train on incomplete trajectories? or trajectories that span episodes? will this mess up the future action prediction since if it spans episodes it may not predict the right actions???
    5)
    """
    action_dim: Sequence[int]
    config: dict

    def setup(self):
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.action_dim, self.config)

    @staticmethod
    def gaussian_sample(mu, log_sigma, key):
        key, _key = jrandom.split(key)  # TODO is this okay idk?
        return mu + jnp.exp(0.5 * log_sigma) * jrandom.normal(_key, mu.shape)

    @nn.compact
    def __call__(self, hidden_encoder, hidden_decoder, past_traj, key, full_traj=False):
        # run the encoder on trajectory
        hidden_encoder, mu, log_sigma = self.encoder(hidden_encoder, past_traj, full_traj)

        latent_sample = self.gaussian_sample(mu, log_sigma, key)

        # run decoder on potential future steps, up to H
        # for agent in range(self.num_agents):
        hidden_decoder, future_actions = self.decoder(hidden_decoder, past_traj[-1], latent_sample, past_traj[0])
        # TODO sort out what hidden does here, are we feeding in the right or wrong ones to each rnn

        # some output concat
        return future_actions, latent_sample, mu, log_sigma, hidden_encoder, hidden_decoder
