from ml_collections import config_dict


def get_config():
    # PPO and MFOS
    config = config_dict.ConfigDict()
    config.SEED = 42

    config.CNN = False
    # config.CNN = True

    config.DISCRETE = False
    # config.DISCRETE = True

    # config.NUM_INNER_STEPS = 1  # 200#0  # ep rollout length
    # config.NUM_EPISODES = 200000  # 2000  # 5000  # 50000  # number of ep rollouts to run

    config.TOTAL_TIMESTEPS = 25000000
    # TODO add in the above and then each agent has a num_inner_steps to figure out num_episodes, this needs to be
    # apparent in the agent itself as well as the overall training loop outside
    # perhaps by

    config.NUM_ENVS = 64  # 128
    config.NUM_DEVICES = 1

    config.DEEP_SEA_MAP = 20

    # config.WANDB = "disabled"
    config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    # config.AGENT_TYPE = "DDPG"
    # config.AGENT_TYPE = "PPO"
    config.AGENT_TYPE = "SAC"
    # config.AGENT_TYPE = "ERSAC"
    # config.AGENT_TYPE = "BootDQN"
    # config.AGENT_TYPE = "DQN"

    config.AGENT_CONFIG = {}

    return config

# TODO need to clarify, for discrete there is only 1 d but A number of actions, for continuous there are Ad actions with a max and min scale

# TODO need to add can_learn to DQN properly


"""
BELNOAZ LoL
B - Batch size, probably when using replay buffer
E - Number of Episodes
L - Episode Length/NUM_INNER_STEPS
S - Seq length if using trajectory buffer
N - Number of Envs
O - Observation Dim
A - Action Dim
C - Action Choices (mostly for discrete actions basically)
Z - More dimensions when in a list
U - Ensemble num
P - Plus
M - Minus

further maybes
M - Number of Meta Episodes
"""
