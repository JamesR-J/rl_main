import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, kaiming_normal, glorot_normal
from typing import Sequence


# class QNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(4, 32, 8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(3136, 512),
#             nn.ReLU(),
#             nn.Linear(512, env.single_action_space.n),
#         )
#
#     def forward(self, x):
#         return self.network(x / 255.0)


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


# class SoftQNetwork(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#
#         def layer_init(layer, bias_const=0.0):
#             nn.init.kaiming_normal_(layer.weight)
#             torch.nn.init.constant_(layer.bias, bias_const)
#             return layer
#
#         obs_shape = envs.single_observation_space.shape
#         self.conv = nn.Sequential(
#             layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
#             nn.Flatten(),
#         )
#
#         with torch.inference_mode():
#             output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]
#
#         self.fc1 = layer_init(nn.Linear(output_dim, 512))
#         self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))
#
#     def forward(self, x):
#         x = F.relu(self.conv(x / 255.0))
#         x = F.relu(self.fc1(x))
#         q_vals = self.fc_q(x)
#         return q_vals

class SoftQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID",
                     kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(x)

        x = nn.Dense(512, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        q_vals = nn.Dense(self.action_dim, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)

        return q_vals


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                    kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(x)

        x = nn.Dense(512, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        logits = nn.Dense(self.action_dim, kernel_init=kaiming_normal(), bias_init=constant(0.0))(x)

        return logits


class PriorAndNotNN(nn.Module):

    @nn.compact
    def __call__(self, data):
        # takes in s and a
        obs, actions = data

        obs = jnp.transpose(obs, (0, 2, 3, 1))
        obs = obs / (255.0)
        obs = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                      kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)
        obs = nn.relu(obs)
        obs = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID",
                      kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)
        obs = nn.relu(obs)
        obs = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                      kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)
        obs = obs.reshape((obs.shape[0], -1))
        obs = nn.relu(obs)

        obs = nn.Dense(48, kernel_init=kaiming_normal(), bias_init=constant(0.0))(obs)
        actions = nn.Dense(16, kernel_init=kaiming_normal(), bias_init=constant(0.0))(actions)
        # TODO its shape 1,3136 for obs, and 1,1 for actions, will this overpower?
        # TODO the above tries to reweight the values for eachother, see the TODO below in reference

        x = jnp.concatenate([obs, actions], axis=1)

        x = nn.Dense(16, kernel_init=glorot_normal())(x)
        x = nn.elu(x)
        x = nn.Dense(16, kernel_init=glorot_normal())(x)
        x = nn.elu(x)
        x = nn.Dense(1, kernel_init=glorot_normal())(x)

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

