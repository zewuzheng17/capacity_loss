import os

# create log path
def create_path_dict(args):
    # args.abs_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    args.algo_name = "rainbow"
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

