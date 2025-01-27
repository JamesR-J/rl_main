import jax.numpy as jnp
import jax.random as jrandom
import jax
import sys


key = jrandom.PRNGKey(42)
in_array = jnp.expand_dims(jnp.arange(0, 33), axis=-1)
print(in_array.shape)
print(in_array)

def bootstrap_samples(key, data, m):
    n = data.shape[0]
    # Generate an array of shape (m, n) with random indices for bootstrapping
    indices = jrandom.randint(key, shape=(m, n), minval=0, maxval=n)
    # Use the indices to gather the bootstrapped samples
    bootstrapped_data = data[indices]
    return bootstrapped_data

ensemble_num = 10
key, _key = jrandom.split(key)
bootstrapped_versions = bootstrap_samples(key, in_array, ensemble_num)
print(bootstrapped_versions)
print(bootstrapped_versions.shape)





