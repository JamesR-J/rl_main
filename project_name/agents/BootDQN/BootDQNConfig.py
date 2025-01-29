from ml_collections import config_dict


def get_BootDQN_config():
    config = config_dict.ConfigDict()
    config.PRIOR_SCALE = 1.0  # 5.0  # 0.5
    config.LR = 1e-3
    config.ENS_LR = 1e-3
    config.TAU_LR = 1e-3  # 1e-2
    config.GAMMA = 0.99
    config.TD_LAMBDA = 0.8
    config.REWARD_NOISE_SCALE = 0.1  # set in ersac paper
    config.UNCERTAINTY_SCALE = 1.0
    config.MASK_PROB = 0.8  # 0.6
    config.HIDDEN_SIZE = 128
    config.NUM_ENSEMBLE = 10
    config.INIT_TAU = 0.02  # set in ersac paper

    config.LOGSTD_MIN = -1
    config.LOGSTD_MAX = 1

    config.BUFFER_SIZE = 100000  # 1e5
    config.BATCH_SIZE = 128
    config.EPS_START = 1.0
    config.EPS_FINISH = 0.05
    config.EPS_DECAY = 0.1

    config.UPDATE_EPOCHS = 4
    config.TARGET_UPDATE_INTERVAL = 10
    config.TAU = 1.0
    config.GAMMA = 0.96

    return config