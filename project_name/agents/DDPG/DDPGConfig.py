from ml_collections import config_dict


def get_DDPG_config():
    config = config_dict.ConfigDict()
    config.LR_CRITIC = 0.001
    config.LR_ACTOR = 0.001

    config.BUFFER_SIZE = 100000#0
    config.BATCH_SIZE = 32  # 128
    config.EPS_DECAY = 0.9  # 1.1

    config.TARGET_UPDATE_INTERVAL = 1  # 10
    config.TAU = 0.001
    config.GAMMA = 0.99

    config.LEARNING_STARTS = 1000  # does this change depending on episodes?

    return config