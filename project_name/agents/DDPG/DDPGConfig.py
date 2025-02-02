from ml_collections import config_dict


def get_DDPG_config():
    config = config_dict.ConfigDict()

    config.NUM_INNER_STEPS = 1

    config.LR_CRITIC = 0.0003
    config.LR_ACTOR = 0.0003
    config.MAX_GRAD_NORM = 0.5

    config.BUFFER_SIZE = 500000#0
    config.BATCH_SIZE = 64  # 256  # 32  # 128
    config.EPS_DECAY = 0.0001  # 1.1

    config.TARGET_UPDATE_INTERVAL = 1  # 0
    config.TAU = 0.005
    config.GAMMA = 0.99

    config.LEARNING_STARTS = 1000  # does this change depending on episodes?

    config.HUBER_LOSS_PARAM = 0.0  # 1.0

    return config