from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42
    config.LR = 1e-4
    config.GAMMA = 0.995
    config.EPS = 1

    # config.TOTAL_TIMESTEPS = 10000000
    config.NUM_UPDATES = 2500  # 10000
    # config.NUM_STEPS = 128

    config.NUM_ENVS = 1

    config.NUM_ENSEMBLE = 10
    config.RP_NOISE = 0.1
    config.SIGMA_SCALE = 3.0

    config.AUTOTUNE = False
    config.TARGET_ENT_SCALE = 0.89
    config.ALPHA = 0.2

    config.BUFFER_SIZE = 100000
    # config.LEARNING_STARTS = int(config.NUM_UPDATES * 0.2)  # starts policy after 20% of outer loops
    config.TARGET_UPDATE_INTERVAL = 4
    config.TAU = 1.0
    config.REPLAY_PRIORITY_EXP = 1.0
    config.IMPORTANCE_SAMPLING_EXP = 0.995

    config.BATCH_SIZE = 32

    config.WANDB = "disabled"  # "online" if want it to work
    # config.WANDB = "online"

    config.DISABLE_JIT = False

    config.DEEPSEA_SIZE = 25

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    return config
