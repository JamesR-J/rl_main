from ml_collections import config_dict


def get_NAIVE_config():
    config = config_dict.ConfigDict()
    config.LR = 0.025
    config.GAMMA = 0.96
    config.EPS = 1
    config.GRU_HIDDEN_DIM = 16 * 3  # to make it divisible by 3 for MFOS, 256
    config.GAE_LAMBDA = 0.95
    config.UPDATE_EPOCHS = 4
    config.NUM_MINIBATCHES = 8  # TODO make it work for one env and one minibatch
    config.CLIP_EPS = 0.2
    config.VF_COEF = 0.5
    config.ENT_COEF = 0.01
    config.SCALE_CLIP_EPS = True  # False
    config.ANNEAL_LR = True
    config.MAX_GRAD_NORM = 0.5

    config.LATENT_DIM = 2
    config.KL_WEIGHT = 0.01

    return config