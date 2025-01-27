import jax
import jax.numpy as jnp
import jax.random as jrandom


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    """Gaussian kernel with dynamic bandwidth.
    The bandwidth is adjusted dynamically to match median_distance / log(Kx).
    See [2] for more information.
    Args:
        xs(`tf.Tensor`): A tensor of shape (M x N x Kx x D) containing MXN sets of Kx
            particles of dimension D. This is the first kernel argument.
        ys(`tf.Tensor`): A tensor of shape (M x N x Ky x D) containing MXN sets of Kx
            particles of dimension D. This is the second kernel argument.
        h_min(`float`): Minimum bandwidth.
    Returns:
        `dict`: Returned dictionary has two fields:
            'output': A `tf.Tensor` object of shape (M x N x Kx x Ky) representing
                the kernel matrix for inputs `xs` and `ys`.
            'gradient': A 'tf.Tensor` object of shape (M x N x Kx x Ky x D)
                representing the gradient of the kernel with respect to `xs`.
    Reference:
        [2] Qiang Liu,Dilin Wang, "Stein Variational Gradient Descent: A General
            Purpose Bayesian Inference Algorithm," Neural Information Processing
            Systems (NIPS), 2016.
    """
    Kx = xs.shape[-2]
    D = xs.shape[-1]
    Ky = ys.shape[-2]
    D2 = ys.shape[-1]
    assert D == D2

    leading_shape = xs.shape[:-2]

    # Compute the pairwise distances of left and right particles.
    diff = jnp.expand_dims(xs, -2) - jnp.expand_dims(ys, -3)
    # ... x Kx x Ky x D

    dist_sq = jnp.sum(diff**2, axis=-1)  # TODO check this but should be ok
    # ... x Kx x Ky

    # Get median.
    values, _ = jax.lax.top_k(jnp.reshape(dist_sq, (*leading_shape, Kx*Ky)),
        (Kx * Ky // 2 + 1))  # This is exactly true only if Kx*Ky is odd  ... x floor(Ks*Kd/2)

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)

    h = medians_sq / jnp.log(Kx)  # ... (shape)
    h = jnp.maximum(h, h_min)
    h = jax.lax.stop_gradient(h)  # Just in case.
    h_expanded_twice = jnp.expand_dims(jnp.expand_dims(h, -1), -1)
    # ... x 1 x 1

    kappa = jnp.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky

    # Construct the gradient
    h_expanded_thrice = jnp.expand_dims(h_expanded_twice, -1)
    # ... x 1 x 1 x 1
    kappa_expanded = jnp.expand_dims(kappa, -1)  # ... x Kx x Ky x 1

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    # ... x Kx x Ky x D

    return {"output": kappa, "gradient": kappa_grad}