from ml_collections import config_dict


def get_config():
    # PPO and MFOS
    config = config_dict.ConfigDict()
    config.SEED = 42

    config.CNN = False
    # config.CNN = True

    config.DISCRETE = False
    # config.DISCRETE = True

    config.NUM_INNER_STEPS = 200#0  # ep rollout length
    config.NUM_EPISODES = 5000  # number of ep rollouts to run
    config.NUM_ENVS = 1  # 16
    config.NUM_DEVICES = 1

    # config.DEEP_SEA_MAP = 1  # 20

    config.WANDB = "disabled"
    # config.WANDB = "online"

    config.DISABLE_JIT = False
    # config.DISABLE_JIT = True

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    # config.AGENT_TYPE = "DDPG"
    config.AGENT_TYPE = "PPO"

    return config


"""
MELNOA LoL
M - Number of Meta Episodes
E - Number of Episodes
L - Episode Length/NUM_INNER_STEPS
N - Number of Envs
O - Observation Dim
A - Action Dim
Z - More dimensions when in a list
"""
