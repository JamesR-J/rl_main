import jax.numpy as jnp
import jax


# action_space = 4
# obs = 12
# num_envs = 1
#
# actions = jnp.arange(0, action_space, step=1)
# actions = jnp.broadcast_to(jnp.expand_dims(actions, axis=(-2, -1)), (*actions.shape, obs, num_envs))
#
# opp_actions = jnp.arange(0, action_space, step=1)
# opp_actions = jnp.broadcast_to(jnp.expand_dims(opp_actions, axis=(-2, -1)), (*opp_actions.shape, obs, num_envs))
#
# def get_reward_noise(action, opp_action):
#     return action+opp_action
#
# with jax.disable_jit():
#     actions_final = jax.vmap(jax.vmap(get_reward_noise, in_axes=(None, 0)), in_axes=(0, None))(actions, opp_actions)
#
# print("YE")
# print(actions_final.shape)

logits_pi = jax.random.normal(jax.random.PRNGKey(42), (4,)) / 1000
logits_rho = jax.random.normal(jax.random.PRNGKey(44), (4,)) / 1000

pi = jax.nn.softmax(logits_pi)
rho = jax.nn.softmax(logits_rho)

joint_prob = pi * rho
eps = 1e-8
log_joint_prob = jnp.log(joint_prob + eps)

result_1 = -jnp.sum(joint_prob * log_joint_prob)
print(result_1)

log_pi = jax.nn.log_softmax(logits_pi)
log_rho = jax.nn.log_softmax(logits_rho)

result_2 = -jnp.sum(joint_prob * (log_pi + log_rho))
print(result_2)