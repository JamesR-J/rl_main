from ml_collections import config_dict


def get_PR2_config():
    config = config_dict.ConfigDict()
    config.LR = 1e-3
    config.GAMMA = 0.95
    config.BUFFER_SIZE = 100000
    config.BATCH_SIZE = 32
    config.REPLAY_PRIORITY_EXP = 1.0
    config.VALUE_N_PARTICLES = 16
    config.KERNEL_N_PARTICLES = 32  # Think got to be double value_n particles
    config.KERNEL_UPDATE_RATIO = 0.5
    config.ANNEALING = 0.5  # TODO set this inside PR2 as it could be trained technically
    config.U_RANGE = 1.0

    return config