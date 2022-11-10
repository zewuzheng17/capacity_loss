import os
from typing import List, Tuple


# create log path
def create_path_dict(args):
    # args.abs_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    log_name = os.path.join(args.task, args.algo_name)  # , str(args.seed), now
    log_path = os.path.join(args.logdir, log_name)

    path_dict = {
        "base": log_name,
        "buffer": os.path.join(log_path, "buffer"),
        "policy": os.path.join(log_path, "policy"),
        "tensor_log": os.path.join(log_path, "tensor_log"),
        "picture": os.path.join(log_path, "picture"),
        "data": os.path.join(log_path, "data"),
    }

    for keys in path_dict.keys():
        if not os.path.exists(path_dict[keys]) and keys != 'base':
            os.makedirs(path_dict[keys])
    return path_dict


# input data, return a smoothed version
def smooth(data: List[List], moving_rate=(0.5, 0.3, 0.2)) -> List[List]:
    smoothed_data = []
    for i in range(len(data)):
        to_bound = len(data) - i
        if to_bound < len(moving_rate):
            smoothed_data.append(data[i])
        else:
            smoothed_data.append([data[i][0],
                                  data[i][1] * moving_rate[0] + data[i + 1][1] * moving_rate[1] + data[i + 2][1] *
                                  moving_rate[2], data[i][2]])
    return smoothed_data
