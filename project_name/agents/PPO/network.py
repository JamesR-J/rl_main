import sys

import flax.linen as nn
import functools
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Optional
import distrax
from ml_collections import ConfigDict
import jax
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import (Distribution, TransformedDistribution, Independent, Normal)
import chex

tfb = tfp.bijectors


class CNNtoLinear(nn.Module):
    @nn.compact
    def __call__(self, obs):
        flatten_layer = jnp.reshape(obs, (obs.shape[0], obs.shape[1], -1))
        return flatten_layer


class DiscreteActorCritic(nn.Module):
    action_dim: int
    config: ConfigDict
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
            embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            embedding = nn.relu(embedding)

        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        embedding = activation(embedding)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1), actor_mean


# class ContinuousActorCritic(nn.Module):  # TODO change this and remove RNN
#     action_dim: int
#     config: ConfigDict
#     activation: str = "tanh"
#
#     @nn.compact
#     def __call__(self, obs):
#         if self.activation == "relu":
#             activation = nn.relu
#         else:
#             activation = nn.tanh
#
#         if self.config.CNN:
#             embedding = CNNtoLinear()(obs)
#         else:
#             embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
#             embedding = nn.relu(embedding)
#
#         embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
#         embedding = activation(embedding)
#         embedding = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
#         embedding = activation(embedding)
#         actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
#
#         critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
#         critic = activation(critic)
#         critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
#         critic = activation(critic)
#         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
#
#         actor_logstd = self.param("actor_logstd", nn.initializers.zeros,(self.action_dim, ))
#         # TODO is the above okay?
#
#         # def callback(action, state):
#         #     print(action)
#         #     print(state)
#         #     print("NEW ONE")
#         #
#         # jax.experimental.io_callback(callback, None, actor_mean, actor_logstd)
#
#         pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))  # TODO check if above works as well
#
#         return pi, jnp.squeeze(critic, axis=-1), actor_mean


class ContinuousActorCritic(nn.Module):
    action_dim: int
    minimum_action: float
    maximum_action: float
    min_scale: float = 1e-3

    @nn.compact
    def __call__(self, x):
        embedding = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(x))
        embedding = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(embedding))

        critic = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(embedding))
        critic = nn.silu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2.0)))(critic))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0))(critic)

        action_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        action_logstd = jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)) + self.min_scale

        pi = Normal(action_mean, jnp.exp(action_logstd))

        return Independent(AffineTanhTransformedDistribution(pi,
                                                             self.minimum_action,
                                                             self.maximum_action),
                           reinterpreted_batch_ndims=1), jnp.squeeze(critic, axis=-1), action_mean


class AffineTanhTransformedDistribution(TransformedDistribution):
    """Distribution followed by tanh and then affine transformations."""

    def __init__(
        self,
        distribution: Distribution,
        minimum: float,
        maximum: float,
        epsilon: float = 1e-3,
        validate_args: bool = False,
    ) -> None:
        """Initialize the distribution with a tanh and affine bijector.

        Args:
          distribution: The distribution to transform.
          minimum: Lower bound of the target range.
          maximum: Upper bound of the target range.
          epsilon: epsilon value for numerical stability.
            epsilon is used to compute the log of the average probability distribution
            outside the clipping range, i.e. on the interval
            [-inf, atanh(inverse_affine(minimum))] for log_prob_left and
            [atanh(inverse_affine(maximum)), inf] for log_prob_right.
          validate_args: Passed to super class.
        """
        # Calculate scale and shift for the affine transformation to achieve the range
        # [minimum, maximum] after the tanh.
        scale = (maximum - minimum) / 2.0
        shift = (minimum + maximum) / 2.0

        # Chain the bijectors
        joint_bijector = tfb.Chain([tfb.Shift(shift), tfb.Scale(scale), tfb.Tanh()])

        super().__init__(
            distribution=distribution, bijector=joint_bijector, validate_args=validate_args
        )

        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, atanh(inverse_affine(minimum))] for
        # log_prob_left and [atanh(inverse_affine(maximum)), inf] for log_prob_right.
        self._min_threshold = minimum + epsilon
        self._max_threshold = maximum - epsilon
        min_inverse_threshold = self.bijector.inverse(self._min_threshold)
        max_inverse_threshold = self.bijector.inverse(self._max_threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(epsilon)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = self.distribution.log_cdf(min_inverse_threshold) - log_epsilon
        self._log_prob_right = (
            self.distribution.log_survival_function(max_inverse_threshold) - log_epsilon
        )

    def log_prob(self, event: chex.Array) -> chex.Array:
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, self._min_threshold, self._max_threshold)
        return jnp.where(
            event <= self._min_threshold,
            self._log_prob_left,
            jnp.where(event >= self._max_threshold, self._log_prob_right, super().log_prob(event)),
        )

    def mode(self) -> chex.Array:
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed: chex.PRNGKey = None) -> chex.Array:
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0
        )

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes: Any = None) -> Any:
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties