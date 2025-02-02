from ml_collections import config_dict


def get_SAC_config():
    config = config_dict.ConfigDict()

    config.NUM_INNER_STEPS = 1

    config.LR_CRITIC = 0.0003
    config.LR_ACTOR = 0.0003
    config.ALPHA_LR = 0.0003
    config.MAX_GRAD_NORM = 0.5

    config.BUFFER_SIZE = 25000#00
    config.BATCH_SIZE = 64  # 128

    config.TARGET_UPDATE_INTERVAL = 10
    config.TAU = 0.005
    config.GAMMA = 0.99

    config.INIT_ALPHA = 0.1

    config.TARGET_ENTROPY_SCALE = 1.0

    return config