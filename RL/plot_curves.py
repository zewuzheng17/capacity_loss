# __package__ = 'RL.evaluate'
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from util.config_args import get_args
from util.utils import create_path_dict


def plot_fig_mix(base_path, plot_type, plot_num, seed_num, xlabel, ylabel, hue, title, save_destination):
    ## load training data
    sns.set()
    fig = plt.figure()

    data = {}
    for i in range(plot_num):
        data[i] = []
        for j in range(seed_num):
            path = os.path.join(base_path, plot_type + "_i" + str(i) + "_s" + str(j)+".pkl")
            with open(path, 'rb') as f:
                files = pickle.load(f)
                data[i] += list(map(lambda x: list(x)+[hue[i]], files)) # hue[i]

    data_total = reduce(lambda x,y: data[x]+data[y], list(data.keys()))
    df = pd.DataFrame(data_total, columns=[xlabel, ylabel, "type"])
    sns.lineplot(x=xlabel, y=ylabel, data=df ,hue="type")  # ,hue="network type"
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    # if plot_type == "curves_loss":
    #     plt.ylim(0, 2e-5)
    plt.title = title
    plt.legend(loc="upper left", fontsize="x-small")
    fig.savefig(save_destination)

def plot_fig_single(base_path, plot_type, plot_id, seed_num, xlabel, ylabel, hue, title, save_destination):
    sns.set()
    fig = plt.figure()

    data = []
    for i in range(seed_num):
        path = os.path.join(base_path, plot_type + "_i" + str(plot_id) + "_s" + str(i) + ".pkl")
        with open(path, 'rb') as f:
            files = pickle.load(f)
            data += list(map(lambda x: list(x)+[hue[plot_id]], files)) # hue[i]

    df = pd.DataFrame(data, columns=[xlabel, ylabel, "type"])
    sns.lineplot(x=xlabel, y=ylabel, data=df ,hue="type")  # ,hue="network type"
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    if plot_type == "curves_loss":
        plt.ylim(0, 2e-5)
    plt.title = title
    plt.legend(loc="upper left", fontsize="x-small")
    fig.savefig(save_destination)


if __name__ == "__main__":
    args = get_args()
    plot_id = int(args.add_infer)
    path_dict = create_path_dict(args)
    plot_num = 2
    seed_num = 1

    if plot_num > 1:
        plot_fig_mix(base_path = path_dict['data'], plot_type="curves_loss", plot_num = plot_num, seed_num = seed_num, xlabel='epoch', \
                     ylabel = 'MSE', hue = ["rainbow", "rainbow + infer"], title="Mean Squared error through training epoch" \
                 , save_destination=os.path.join(path_dict['picture'],"loss_curves.png"))

        plot_fig_mix(base_path = path_dict['data'], plot_type="curves_EFdimension", plot_num = plot_num, seed_num = seed_num,xlabel="epoch", \
                 ylabel="Effective dimension", hue = ["rainbow", "rainbow + infer"], title="Feature rank through training epoch" \
                 , save_destination=os.path.join(path_dict['picture'],"Efdimension_curves.png"))

        plot_fig_mix(base_path = path_dict['data'], plot_type="rewards", plot_num = plot_num, seed_num = seed_num,xlabel="epoch", \
                 ylabel="return", hue = ["rainbow", "rainbow + infer"], title="rewards through training epoch" \
                 , save_destination=os.path.join(path_dict['picture'],"reward_curves.png"))

    else:
        plot_fig_single(base_path=path_dict['data'], plot_type="curves_loss", plot_id=plot_id, seed_num=seed_num,
                     xlabel='epoch', \
                     ylabel='MSE', hue=["rainbow", "rainbow + infer"], title="Mean Squared error through training epoch" \
                     , save_destination=os.path.join(path_dict['picture'], "loss_curves_i{}.png".format(plot_id)))

        plot_fig_single(base_path=path_dict['data'], plot_type="curves_EFdimension", plot_id=plot_id, seed_num=seed_num,
                     xlabel="epoch", \
                     ylabel="Effective dimension", hue=["rainbow", "rainbow + infer"],
                     title="Feature rank through training epoch" \
                     , save_destination=os.path.join(path_dict['picture'], "Efdimension_curves_i{}.png".format(plot_id)))

        plot_fig_single(base_path=path_dict['data'], plot_type="rewards", plot_id=plot_id, seed_num=seed_num, xlabel="epoch", \
                 ylabel="return", hue=["rainbow", "rainbow + infer"], title="rewards through training epoch" \
                 , save_destination=os.path.join(path_dict['picture'], "reward_curves_i{}.png".format(plot_id)))

#os.system('rm -rf ./data/Store_datatype*.pickle')
