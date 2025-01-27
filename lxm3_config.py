import sys
import itertools
from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()


    return config  # TODO get this to work at some point


def sweep_SWEEP():
    seed_list = [28, 10, 98, 44, 22, 68]
    depth = [25, 30, 35, 40, 45, 50]
    depth = [15]

    combinations = itertools.product(seed_list, depth)
    result = [{"seed": seed,
               "deepsea_depth": depth,
               # "homogeneous": homo,
               # "num_agents": agent,
               # "reward_type": reward,
               # "split_train": False,
               # "num_loops": num_loops,
               "disable_jit": False} for seed, depth in combinations]

    return result

# def get_sweep():
#     """Returns a sweep configuration for hyperparameter tuning."""
#
#     config = config_dict.ConfigDict()
#     config.learning_rate = 0.001
#     sweep_config.params["batch_size"] = config_dict.randint(16, 64)
#     # Add other hyperparameters with sweep ranges
#     return sweep
