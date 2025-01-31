import sys

import chex
import jax
import flax.linen as nn
import functools
from functools import partial
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Callable
import distrax
from ml_collections import ConfigDict
from tensorflow_probability.substrates.jax.distributions import Deterministic, Distribution

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


class ContinuousQNetwork(nn.Module):  # TODO change this and remove RNN
    config: ConfigDict
    activation: str = "tanh"
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, obs, actions):
        # if self.config.CNN:  # TODO turned off CNN as well
        #     embedding = CNNtoLinear()(obs)
        # else:
        #     embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        #     embedding = nn.relu(embedding)

        # s1 = nn.swish(nn.Dense(256, kernel_init=nn.initializers.kaiming_uniform())(obs))
        # s2 = nn.swish(nn.Dense(128, kernel_init=nn.initializers.kaiming_uniform())(s1))
        # a1 = nn.swish(nn.Dense(128, kernel_init=nn.initializers.kaiming_uniform())(actions))
        #
        # new_x = jnp.concatenate((s2, a1), axis=-1)
        # new_x = nn.swish(nn.Dense(128, kernel_init=nn.initializers.kaiming_uniform())(new_x))
        # q_vals = nn.Dense(1, kernel_init=nn.initializers.uniform(0.003))(new_x)

        x = jnp.concatenate((obs, actions), axis=-1)
        x = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(x))
        x = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(x))
        # x = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(x))
        # x = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(x))
        q_vals = nn.Dense(1, kernel_init=orthogonal(1.0))(x)

        return q_vals


class DeterministicPolicy(nn.Module):
    action_dim: int
    action_min: float
    action_max: float

    @nn.compact
    def __call__(self, obs):
        # if self.config.CNN:  # TODO turned off CNN as well
        #     embedding = CNNtoLinear()(obs)
        # else:
        #     embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        #     embedding = nn.relu(embedding)

        x = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(obs))
        x = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(x))
        # x = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(x))
        # x = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(x))

        action = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)

        action = Deterministic(action)

        return ScalePostProcessor(minimum=self.action_min, maximum=self.action_max, scale_fn=tanh_to_spec)(action)


def tanh_to_spec(inputs: chex.Array, minimum: float, maximum: float) -> chex.Array:
    scale = maximum - minimum
    offset = minimum
    inputs = jax.nn.tanh(inputs)  # [-1, 1]
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * scale + offset  # [minimum, maximum]
    # output = inputs
    return output


class ScalePostProcessor(nn.Module):
    minimum: float
    maximum: float
    scale_fn: Callable[[chex.Array, float, float], chex.Array]

    @nn.compact
    def __call__(self, distribution: Distribution) -> Distribution:
        post_processor = partial(
            self.scale_fn, minimum=self.minimum, maximum=self.maximum
        )  # type: ignore
        return PostProcessedDistribution(distribution, post_processor)


class PostProcessedDistribution(Distribution):
    """A distribution that applies a postprocessing function to the samples and mode.

    This is useful for transforming the output of a distribution to a different space, such as
    rescaling the output of a tanh-transformed Normal distribution to a different range. However,
    this is not the same as a bijector, which also transforms the density function of the
    distribution. This is only useful for transforming the samples and mode of the distribution.
    For example, for an algorithm that requires taking the log probability of the samples, the
    distribution should be transformed using a bijector, not a postprocessor."""

    def __init__(
        self, distribution: Distribution, postprocessor: Callable[[chex.Array], chex.Array]
    ):
        self.distribution = distribution
        self.postprocessor = postprocessor

    def sample(self, seed: chex.PRNGKey, sample_shape: Sequence[int] = ()) -> chex.Array:
        return self.postprocessor(self.distribution.sample(seed=seed, sample_shape=sample_shape))

    def mode(self) -> chex.Array:
        return self.postprocessor(self.distribution.mode())

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.distribution, name)