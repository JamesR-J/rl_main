from ml_collections import config_dict


def get_SAC_config():
    config = config_dict.ConfigDict()
    config.LR = 0.001
    config.EPS = 1
    config.GRU_HIDDEN_DIM = 16
    config.GAE_LAMBDA = 0.95
    config.NUM_MINIBATCHES = 4

    config.BUFFER_SIZE = 10000#0  # 1e5  # TODO change back to full asap
    config.BATCH_SIZE = 32  # 128
    config.EPS_START = 1.0
    config.EPS_FINISH = 0.05
    config.EPS_DECAY = 0.1

    config.UPDATE_EPOCHS = 4
    config.TARGET_UPDATE_INTERVAL = 10
    config.TAU = 0.001
    config.GAMMA = 0.99

    config.LEARNING_STARTS = 1000  # does this change depending on episodes?

    config.ACTION_SCALE = 1.0
    config.EXPLORATION_NOISE = 0.1  # 0.2

    config.LOGSTD_MIN = -1
    config.LOGSTD_MAX = 1
    config.INIT_ALPHA = 0.1
    config.ALPHA_LR = 0.001

    return config