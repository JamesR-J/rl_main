from ml_collections import config_dict


def get_DDPG_config():
    config = config_dict.ConfigDict()
    config.LR_CRITIC = 0.001
    config.LR_ACTOR = 0.001
    config.EPS = 1
    config.GRU_HIDDEN_DIM = 16
    config.GAE_LAMBDA = 0.95
    config.NUM_MINIBATCHES = 4

    config.BUFFER_SIZE = 10000#0  # 1e5  # TODO change back to full asap
    config.BATCH_SIZE = 128
    config.EPS_START = 1.0
    config.EPS_FINISH = 0.05
    config.EPS_DECAY = 0.1

    config.UPDATE_EPOCHS = 4
    config.TARGET_UPDATE_INTERVAL = 10
    config.TAU = 0.001
    config.GAMMA = 0.99

    config.LEARNING_STARTS = 1000  # does this change depending on episodes?

    config.ACTION_SCALE = 1.0
    config.EXPLORATION_NOISE = 0.1

    return config