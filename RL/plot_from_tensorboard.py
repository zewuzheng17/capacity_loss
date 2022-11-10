from tensorboard.backend.event_processing import event_accumulator
from util.config_args_mujoco import get_args
from util.utils import create_path_dict, smooth
from typing import Dict, List, Tuple
import re

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

args = get_args()
base_path = create_path_dict(args)
tensor_path = base_path['tensor_log']
picture_path = base_path['picture']

data_path_dict = {
    "SAC, RR=1": ["False0", "False1", "False2"],
    "SAC, RR=9": ["False3", "False4", "False5"],
    "SAC + reset, RR=1": ["True0", "True1", "True2"],
    "SAC + reset, RR=9": ["True3", "True4", "True5"]
}

# data_path_dict = {
#     "SAC, RR=1": ["False0", "False1", "False2"],
#     "SAC + reset, RR=1": ["True0", "True1", "True2"],
# }

plot_type_list = ['update/loss/ratio', 'train/reward', 'capacity/abs_dimension_p',
                  'capacity/abs_dimension_q',
                  'capacity/percent_zero_weights_p', 'capacity/percent_zero_weights_q']   # 'update/loss/critic1', 'update/loss/critic1_test',,,

"""
    input the path that contains tensor log, and subfolder,
    output a dict containing path to load tensorboard files from.
"""


def get_latest_file(path: str, path_dict: Dict) -> Dict:
    output_dict = {}
    for keys in path_dict.keys():
        whole_data = []
        for folder in path_dict[keys]:
            base_folder = os.path.join(path, folder)
            list_files = os.listdir(base_folder)
            list_files.sort(key=lambda fn: os.path.getmtime(base_folder + "/" + fn))
            latest_file = os.path.join(base_folder, list_files[-1])
            whole_data.append(latest_file)
        output_dict.update({keys: whole_data})
    return output_dict


def read_tensorboard_data(path_dict: Dict, vals: str) -> List[List]:
    whole_data = []
    for keys in path_dict.keys():
        for paths in path_dict[keys]:
            data = event_accumulator.EventAccumulator(paths)
            data.Reload()
            data_scalar = data.scalars.Items(vals)
            whole_data += smooth(list(map(lambda x: list(x)[1:] + [keys], data_scalar)))
    return whole_data


paths_dict = get_latest_file(tensor_path, data_path_dict)
dash_dict = {
    "SAC, RR=1": (2, 2),
    "SAC, RR=9": (2, 2),
    "SAC + reset, RR=1": (1, 0),
    "SAC + reset, RR=9": (1, 0)
}
for plot_type in plot_type_list:
    print("plotting {}".format(plot_type))
    color = ['blue', 'green', 'orange', 'red']
    data_plot_type = read_tensorboard_data(paths_dict, plot_type)
    sns.set()
    # sns.set_palette(sns.color_palette("husl", 4))
    sns.set_palette(sns.color_palette(color))
    fig = plt.figure()
    df = pd.DataFrame(data_plot_type, columns=["steps", plot_type.replace("/", " "), "types"])
    if re.search("ratio", plot_type):
        df = df[(df['types'] == "SAC, RR=9") | (df['types'] == "SAC, RR=1")]  #
        df['steps'][df['types'] == "SAC, RR=9"] /= 9
    sns.lineplot(x="steps", y=plot_type.replace("/", " "), data=df, hue="types", style="types", dashes=dash_dict)
    plt.title("plots for {}".format(plot_type.replace("/", " ")))
    if re.search("ratio", plot_type):
        plt.ylim(0, 10)
    plt.legend(loc="upper left", fontsize="x-small")
    print("saving plots")
    fig.savefig(os.path.join(picture_path, "{}.png".format(plot_type.replace("/", "_"))))
