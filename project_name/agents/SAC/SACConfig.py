from ml_collections import config_dict


def get_SAC_config():
    config = config_dict.ConfigDict()
    config.LR = 0.0003
    config.EPS = 1

    config.BUFFER_SIZE = 100000#0
    config.BATCH_SIZE = 64  # 128

    config.TARGET_UPDATE_INTERVAL = 10
    config.TAU = 0.005
    config.GAMMA = 0.99

    config.INIT_ALPHA = 0.1
    config.ALPHA_LR = 0.001

    return config