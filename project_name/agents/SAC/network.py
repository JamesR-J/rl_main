import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, kaiming_normal, glorot_normal
from typing import Sequence
import sys
from ml_collections import ConfigDict


class SoftQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(256, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x))
        x = nn.relu(nn.Dense(128, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x))
        q_vals = nn.Dense(self.action_dim, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)

        return q_vals

class DiscreteDoubleSoftQNetwork(nn.Module):
    action_dim: int

    def setup(self):
        self.qf_1 = SoftQNetwork(self.action_dim)
        self.qf_2 = SoftQNetwork(self.action_dim)

    def __call__(self, x):
        qf1 = self.qf_1(x)
        qf2 = self.qf_2(x)

        qf_result = jnp.concatenate((jnp.expand_dims(qf1, axis=-1), jnp.expand_dims(qf2, axis=-1)), axis=-1)

        return qf_result


class ContinuousDoubleSoftQNetwork(nn.Module):
    def setup(self):
        self.qf_1 = SoftQNetwork(1)
        self.qf_2 = SoftQNetwork(1)

    def __call__(self, s, a):
        x = jnp.concatenate((s, a), axis=-1)
        qf1 = self.qf_1(x)
        qf2 = self.qf_2(x)

        qf_result = jnp.concatenate((jnp.expand_dims(qf1, axis=-1), jnp.expand_dims(qf2, axis=-1)), axis=-1)

        return qf_result


class DiscreteActor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(256, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x))
        x = nn.relu(nn.Dense(128, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x))

        logits = nn.Dense(self.action_dim, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)

        pi = distrax.Categorical(logits=logits)

        return pi


class ContinuousActor(nn.Module):
    action_dim: int
    agent_config: ConfigDict

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(256, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x))
        x = nn.relu(nn.Dense(128, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x))

        action_mean = nn.Dense(self.action_dim, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        action_logstd = nn.tanh(nn.Dense(self.action_dim, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x))
        action_logstd = self.agent_config.LOGSTD_MIN + 0.5 * (self.agent_config.LOGSTD_MAX - self.agent_config.LOGSTD_MIN) * (action_logstd + 1)

        pi = distrax.Normal(action_mean, jnp.exp(action_logstd))

        return pi
