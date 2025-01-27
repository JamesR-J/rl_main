from ml_collections import config_dict


def get_IDQN_config():
    config = config_dict.ConfigDict()
    config.LR = 0.005
    config.EPS = 1
    config.GRU_HIDDEN_DIM = 16
    config.GAE_LAMBDA = 0.95
    config.NUM_MINIBATCHES = 4
    config.CLIP_EPS = 0.2
    config.VF_COEF = 0.5
    config.ENT_COEF = 0.01
    config.SCALE_CLIP_EPS = True  # False
    config.ANNEAL_LR = False
    config.MAX_GRAD_NORM = 0.5
    config.ADAM_EPS = 1e-5

    config.BUFFER_SIZE = 100000  # 1e5
    config.BATCH_SIZE = 128
    config.EPS_START = 1.0
    config.EPS_FINISH = 0.05
    config.EPS_DECAY = 0.1

    config.UPDATE_EPOCHS = 4
    config.TARGET_UPDATE_INTERVAL = 10
    config.TAU = 1.0
    config.GAMMA = 0.96

    config.LEARNING_STARTS = 1000  # does this change depending on episodes?

    return config