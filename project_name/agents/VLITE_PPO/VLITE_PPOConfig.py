from ml_collections import config_dict


def get_VLITE_PPO_config():
    config = config_dict.ConfigDict()
    config.PRIOR_SCALE = 1.0  # 5.0  # 0.5
    config.LR = 1e-3
    config.ENS_LR = 1e-3
    config.TD_LAMBDA = 0.8
    config.REWARD_NOISE_SCALE = 0.1  # set in ersac paper
    config.UNCERTAINTY_SCALE = 1.0
    config.MASK_PROB = 0.8  # 0.6
    config.HIDDEN_SIZE = 128
    config.NUM_ENSEMBLE = 10

    config.GAMMA = 0.96
    config.EPS = 1
    config.GRU_HIDDEN_DIM = 16
    config.GAE_LAMBDA = 0.95
    config.UPDATE_EPOCHS = 2
    config.NUM_MINIBATCHES = 4
    config.CLIP_EPS = 0.2
    config.VF_COEF = 0.5
    config.ENT_COEF = 0.01
    config.SCALE_CLIP_EPS = True  # False
    config.ANNEAL_LR = False
    config.MAX_GRAD_NORM = 0.5
    config.ADAM_EPS = 1e-5

    return config