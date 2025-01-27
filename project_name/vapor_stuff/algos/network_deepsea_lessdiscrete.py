import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, kaiming_normal, glorot_normal
from typing import Sequence
import sys


class SoftQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x, a):
        x = nn.Conv(32, kernel_size=(2, 2), strides=(1, 1), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        x = nn.Conv(64, kernel_size=(2, 2), strides=(1, 1), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(127)(x)
        concat_obs = jnp.concatenate((x, a), axis=-1)

        embedding = nn.Dense(256, kernel_init=kaiming_normal(), bias_init=constant(0.0))(concat_obs)
        embedding = nn.relu(embedding)

        embedding = nn.Dense(128, kernel_init=kaiming_normal(), bias_init=constant(0.0))(embedding)
        embedding = nn.relu(embedding)

        q_vals = nn.Dense(1, kernel_init=kaiming_normal(), bias_init=constant(0.0))(embedding)

        return q_vals


class DoubleSoftQNetwork(nn.Module):
    action_dim: int

    def setup(self):
        self.qf_1 = SoftQNetwork(self.action_dim)
        self.qf_2 = SoftQNetwork(self.action_dim)

    def __call__(self, x, a):
        qf1 = self.qf_1(x, a)
        qf2 = self.qf_2(x, a)

        qf_result = jnp.concatenate((jnp.expand_dims(qf1, axis=-1), jnp.expand_dims(qf2, axis=-1)), axis=-1)

        return qf_result


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, kernel_size=(2, 2), strides=(1, 1), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        x = nn.Conv(64, kernel_size=(2, 2), strides=(1, 1), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        x = nn.Dense(128, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        logits = nn.Dense(self.action_dim, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)

        # pi_s = nn.softmax(logits, axis=1)
        # log_pi_s = jnp.log(pi_s + (pi_s == 0) * 1e-8)
        #
        # return pi_s, log_pi_s

        return logits


class PriorAndNotNN(nn.Module):

    @nn.compact
    def __call__(self, data):
        # takes in s and a
        obs, actions = data

        obs = nn.Conv(32, kernel_size=(2, 2), strides=(1, 1), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)
        obs = nn.relu(obs)

        obs = nn.Conv(64, kernel_size=(2, 2), strides=(1, 1), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)

        obs = obs.reshape((obs.shape[0], -1))
        obs = nn.Dense(256, kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)
        obs = nn.relu(obs)

        obs = nn.Dense(128, kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)
        obs = nn.relu(obs)

        obs = nn.Dense(48, kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)
        actions = nn.Dense(16, kernel_init=kaiming_normal(), bias_init=constant(0.0))(actions)
        # TODO its shape 1,3136 for obs, and 1,1 for actions, will this overpower?
        # TODO the above tries to reweight the values for eachother, see the TODO below in reference

        x = jnp.concatenate([obs, actions], axis=1)

        x = nn.Dense(64, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(128, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)

        # TODO test to do, want 0 mean and unit std deve after each linear layer of activation, check activation post each layer and print
        # TODO the x mean and x std and check that is 0 and unit std

        return x


class RandomisedPrior(nn.Module):
    static_prior: PriorAndNotNN = PriorAndNotNN()
    trainable: PriorAndNotNN = PriorAndNotNN()
    beta: float = 3

    @nn.compact
    def __call__(self, x):
        x1 = self.static_prior(x)
        x2 = self.trainable(x)

        return self.beta * x1 + x2

