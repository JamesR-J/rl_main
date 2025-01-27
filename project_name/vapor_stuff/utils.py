import chex
from typing import Optional
import jax.numpy as jnp
import chex
import distrax
import collections
import jax
# from rlax._src import distributions
from distrax._src.utils import math
import jax.random as jrandom


@chex.dataclass(frozen=True)  # TODO is in here and algo so should add both to utils
class TransitionNoInfo:
    state: chex.Array
    action: chex.Array
    reward: chex.Array
    ensemble_reward: chex.Array
    done: chex.Array
    logits: chex.Array


Array = chex.Array
Numeric = chex.Numeric
VTraceOutput = collections.namedtuple(
    'vtrace_output', ['errors', 'pg_advantage', 'q_estimate'])


def entropy_loss_fn(logits_t, uncertainty_t, mask):
    log_pi = jax.nn.log_softmax(logits_t)
    log_pi_pi = math.mul_exp(log_pi, log_pi)
    entropy_per_timestep = -jnp.sum(log_pi_pi * uncertainty_t, axis=-1)
    return -jnp.mean(entropy_per_timestep * mask)


def policy_gradient_loss(
        logits_t: Array,
        a_t: Array,
        adv_t: Array,
        w_t: Array,
        use_stop_gradient: bool = True,
) -> Array:
    """Calculates the policy gradient loss.

  See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
  (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

  Args:
    logits_t: a sequence of unnormalized action preferences.
    a_t: a sequence of actions sampled from the preferences `logits_t`.
    adv_t: the observed or estimated advantages from executing actions `a_t`.
    w_t: a per timestep weighting for the loss.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.

  Returns:
    Loss whose gradient corresponds to a policy gradient update.
  """
    chex.assert_rank([logits_t, a_t, adv_t, w_t], [2, 1, 1, 1])
    chex.assert_type([logits_t, a_t, adv_t, w_t], [float, int, float, float])

    log_pi_a_t = distrax.Softmax(logits_t).log_prob(a_t)
    adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
    loss_per_timestep = -log_pi_a_t * adv_t
    return jnp.mean(loss_per_timestep * w_t)


# Generate bootstrap of given size
def generate_bootstrap(key, size):
    seed, _ = jax.random.split(key)
    return [jax.random.randint(seed + i, (), 0, size) for i in range(size)]


def bootstrap_samples(key, action, obs, reward=None, m=10):
    n = action.shape[0]
    # Generate an array of shape (m, n) with random indices for bootstrapping
    indices = jrandom.randint(key, shape=(m, n), minval=0, maxval=n)
    # Use the indices to gather the bootstrapped samples
    bootstrapped_action = action[indices]
    bootstrapped_obs = obs[indices]
    if reward is not None:
        bootstrapped_reward = reward[indices]
    else:
        bootstrapped_reward = None
    return bootstrapped_action, bootstrapped_obs, bootstrapped_reward


def l2_loss(predictions: Array,
            targets: Optional[Array] = None) -> Array:
    """Caculates the L2 loss of predictions wrt targets.
  FROM RLAX BUT COPIED HERE

  If targets are not provided this function acts as an L2-regularizer for preds.

  Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
    if targets is None:
        targets = jnp.zeros_like(predictions)
    chex.assert_type([predictions, targets], float)
    return 0.5 * (predictions - targets) ** 2


def categorical_importance_sampling_ratios(pi_logits_t: Array,
                                           mu_logits_t: Array,
                                           a_t: Array) -> Array:
    """Compute importance sampling ratios from logits.

  Args:
    pi_logits_t: unnormalized logits at time t for the target policy.
    mu_logits_t: unnormalized logits at time t for the behavior policy.
    a_t: actions at time t.

  Returns:
    importance sampling ratios.
  """
    # warnings.warn(
    #     "Rlax categorical_importance_sampling_ratios will be deprecated. "
    #     "Please use distrax.importance_sampling_ratios instead.",
    #     PendingDeprecationWarning, stacklevel=2
    # )
    return distrax.importance_sampling_ratios(distrax.Categorical(
        pi_logits_t), distrax.Categorical(mu_logits_t), a_t)


def entropy_loss(
        logits_t: Array,
        w_t: Array,
) -> Array:
    """Calculates the entropy regularization loss.

  See "Function Optimization using Connectionist RL Algorithms" by Williams.
  (https://www.tandfonline.com/doi/abs/10.1080/09540099108946587)

  Args:
    logits_t: a sequence of unnormalized action preferences.
    w_t: a per timestep weighting for the loss.

  Returns:
    Entropy loss.
  """
    chex.assert_rank([logits_t, w_t], [2, 1])
    chex.assert_type([logits_t, w_t], float)

    entropy_per_timestep = distrax.Softmax(logits_t).entropy()
    return -jnp.mean(entropy_per_timestep * w_t)


def vtrace(
        v_tm1: Array,
        v_t: Array,
        r_t: Array,
        discount_t: Array,
        rho_tm1: Array,
        lambda_: Numeric = 1.0,
        clip_rho_threshold: float = 1.0,
        stop_target_gradients: bool = True,
) -> Array:
    """Calculates V-Trace errors from importance weights.

  V-trace computes TD-errors from multistep trajectories by applying
  off-policy corrections based on clipped importance sampling ratios.

  See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
  Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561).

  Args:
    v_tm1: values at time t-1.
    v_t: values at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    rho_tm1: importance sampling ratios at time t-1.
    lambda_: mixing parameter; a scalar or a vector for timesteps t.
    clip_rho_threshold: clip threshold for importance weights.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    V-Trace error.
  """
    chex.assert_rank([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                     [1, 1, 1, 1, 1, {0, 1}])
    chex.assert_type([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                     [float, float, float, float, float, float])
    chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

    # Clip importance sampling ratios.
    c_tm1 = jnp.minimum(1.0, rho_tm1) * lambda_
    clipped_rhos_tm1 = jnp.minimum(clip_rho_threshold, rho_tm1)

    # Compute the temporal difference errors.
    td_errors = clipped_rhos_tm1 * (r_t + discount_t * v_t - v_tm1)

    # Work backwards computing the td-errors.
    def _body(acc, xs):
        td_error, discount, c = xs
        acc = td_error + discount * c * acc
        return acc, acc

    _, errors = jax.lax.scan(
        _body, 0.0, (td_errors, discount_t, c_tm1), reverse=True)

    # Return errors, maybe disabling gradient flow through bootstrap targets.
    return jax.lax.select(
        stop_target_gradients,
        jax.lax.stop_gradient(errors + v_tm1) - v_tm1,
        errors)


def vtrace_td_error_and_advantage(
        v_tm1: Array,
        v_t: Array,
        r_t: Array,
        discount_t: Array,
        rho_tm1: Array,
        lambda_: Numeric = 1.0,
        clip_rho_threshold: float = 1.0,
        clip_pg_rho_threshold: float = 1.0,
        stop_target_gradients: bool = True,
) -> VTraceOutput:
    """Calculates V-Trace errors and PG advantage from importance weights.

  This functions computes the TD-errors and policy gradient Advantage terms
  as used by the IMPALA distributed actor-critic agent.

  See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
  Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561)

  Args:
    v_tm1: values at time t-1.
    v_t: values at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    rho_tm1: importance weights at time t-1.
    lambda_: mixing parameter; a scalar or a vector for timesteps t.
    clip_rho_threshold: clip threshold for importance ratios.
    clip_pg_rho_threshold: clip threshold for policy gradient importance ratios.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    a tuple of V-Trace error, policy gradient advantage, and estimated Q-values.
  """
    chex.assert_rank([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                     [1, 1, 1, 1, 1, {0, 1}])
    chex.assert_type([v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_],
                     [float, float, float, float, float, float])
    chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_tm1])

    # If scalar make into vector.
    lambda_ = jnp.ones_like(discount_t) * lambda_

    errors = vtrace(
        v_tm1, v_t, r_t, discount_t, rho_tm1,
        lambda_, clip_rho_threshold, stop_target_gradients)
    targets_tm1 = errors + v_tm1
    q_bootstrap = jnp.concatenate([
        lambda_[:-1] * targets_tm1[1:] + (1 - lambda_[:-1]) * v_tm1[1:],
        v_t[-1:],
    ], axis=0)
    q_estimate = r_t + discount_t * q_bootstrap
    clipped_pg_rho_tm1 = jnp.minimum(clip_pg_rho_threshold, rho_tm1)
    pg_advantages = clipped_pg_rho_tm1 * (q_estimate - v_tm1)
    return VTraceOutput(
        errors=errors, pg_advantage=pg_advantages, q_estimate=q_estimate)
