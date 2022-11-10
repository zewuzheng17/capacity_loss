# __package__ = 'RL.evaluate'
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from scipy import stats
from functools import reduce
from util.config_args import get_args
from util.utils import create_path_dict


def plot_fig_mix(base_path, plot_type, plot_num, seed_num, xlabel, ylabel, hue, title, save_destination, args):
    ## load training data
    sns.set()
    fig = plt.figure()

    data = {}
    for i in plot_num:
        data[i] = []
        for j in seed_num:
            path = os.path.join(base_path, plot_type + "_i" + str(i) + "_s" + str(j) + ".pkl")
            with open(path, 'rb') as f:
                files = pickle.load(f)
                if plot_type == "rewards":
                    data[i] += list(
                        map(lambda x: list(x) + [hue[i]], files[:int(args.test_capacity_length)]))  # hue[i]
                else:
                    data[i] += list(map(lambda x: list(x) + [hue[i]], files))

    data_total = reduce(lambda x, y: data[x] + data[y], list(data.keys()))

    if plot_type == "curves_loss":
        df = pd.DataFrame(data_total, columns=[xlabel, ylabel, "initial_loss", "type"])
        q = df[ylabel].quantile(0.95)
        df = df[df[ylabel] < q]
        if re.search("initial", save_destination):
            sns.lineplot(x=xlabel, y="initial_loss", data=df, hue="type", linestyle='--')
    else:
        df = pd.DataFrame(data_total, columns=[xlabel, ylabel, "type"])

    sns.lineplot(x=xlabel, y=ylabel, data=df, hue="type")  # ,hue="network type"
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.title = title
    if plot_type == "curves_EFdimension":
        plt.ylim((300, 370))
    plt.legend(loc="upper left", fontsize="x-small")
    fig.savefig(save_destination)


def plot_fig_single(base_path, plot_type, plot_id, seed_num, xlabel, ylabel, hue, title, save_destination, args):
    sns.set()
    fig = plt.figure()

    data = []
    for i in seed_num:  # range(seed_num):
        path = os.path.join(base_path, plot_type + "_i" + str(plot_id) + "_s" + str(i) + ".pkl")
        with open(path, 'rb') as f:
            files = pickle.load(f)
            if plot_type == "rewards":
                data += list(map(lambda x: list(x) + [hue[plot_id]], files[:args.test_capacity_length]))  # hue[i]
            else:
                data += list(map(lambda x: list(x) + [hue[plot_id]], files))
    if plot_type == "curves_loss":
        df = pd.DataFrame(data, columns=[xlabel, ylabel, "initial_loss", "type"])
        q = df[ylabel].quantile(0.95)
        df = df[df[ylabel] < q]
        if re.search("initial", save_destination):
            sns.lineplot(x=xlabel, y="initial_loss", data=df, hue="type", linestyle='--')
    else:
        df = pd.DataFrame(data, columns=[xlabel, ylabel, "type"])

    sns.lineplot(x=xlabel, y=ylabel, data=df, hue="type")  # ,hue="network type"
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.title = title
    plt.legend(loc="upper left", fontsize="x-small")
    fig.savefig(save_destination)


if __name__ == "__main__":
    args = get_args()
    args.algo_name = 'rainbow'
    plot_id = int(args.add_infer)
    path_dict = create_path_dict(args)
    plot_num = [0, 1]  # plot type, add infer = 1, not add infer = 0
    seed_num = [0, 1, 2]  # indentify which seeds to plot

    if len(plot_num) > 1:
        plot_fig_mix(base_path=path_dict['data'], plot_type="curves_loss", plot_num=plot_num, seed_num=seed_num,
                     xlabel='epoch', \
                     ylabel='MSE', hue=["rainbow", "rainbow + infer"], title="Mean Squared error through training epoch" \
                     , save_destination=os.path.join(path_dict['picture'], "loss_curves.png"), args=args)

        plot_fig_mix(base_path=path_dict['data'], plot_type="curves_loss", plot_num=plot_num, seed_num=seed_num,
                     xlabel='epoch', \
                     ylabel='MSE', hue=["rainbow", "rainbow + infer"], title="Mean Squared error through training epoch" \
                     , save_destination=os.path.join(path_dict['picture'], "loss_curves_with_initial.png"), args=args)

        plot_fig_mix(base_path=path_dict['data'], plot_type="curves_EFdimension", plot_num=plot_num, seed_num=seed_num,
                     xlabel="epoch", \
                     ylabel="Effective dimension", hue=["rainbow", "rainbow + infer"],
                     title="Feature rank through training epoch" \
                     , save_destination=os.path.join(path_dict['picture'], "Efdimension_curves.png"), args=args)

        plot_fig_mix(base_path=path_dict['data'], plot_type="rewards", plot_num=plot_num, seed_num=seed_num,
                     xlabel="epoch", \
                     ylabel="return", hue=["rainbow", "rainbow + infer"], title="rewards through training epoch" \
                     , save_destination=os.path.join(path_dict['picture'], "reward_curves.png"), args=args)

    else:
        plot_fig_single(base_path=path_dict['data'], plot_type="curves_loss", plot_id=plot_id, seed_num=seed_num,
                        xlabel='epoch', \
                        ylabel='MSE', hue=["rainbow", "rainbow + infer"],
                        title="Mean Squared error through training epoch" \
                        , save_destination=os.path.join(path_dict['picture'], "loss_curves_i{}.png".format(plot_id)),
                        args=args)

        plot_fig_single(base_path=path_dict['data'], plot_type="curves_loss", plot_id=plot_id, seed_num=seed_num,
                        xlabel='epoch', \
                        ylabel='MSE', hue=["rainbow", "rainbow + infer"],
                        title="Mean Squared error through training epoch" \
                        , save_destination=os.path.join(path_dict['picture'],
                                                        "loss_curves_with_initial_i{}.png".format(plot_id)), args=args)

        plot_fig_single(base_path=path_dict['data'], plot_type="curves_EFdimension", plot_id=plot_id, seed_num=seed_num,
                        xlabel="epoch", \
                        ylabel="Effective dimension", hue=["rainbow", "rainbow + infer"],
                        title="Feature rank through training epoch" \
                        , save_destination=os.path.join(path_dict['picture'],
                                                        "Efdimension_curves_i{}.png".format(plot_id)), args=args)

        plot_fig_single(base_path=path_dict['data'], plot_type="rewards", plot_id=plot_id, seed_num=seed_num,
                        xlabel="epoch", \
                        ylabel="return", hue=["rainbow", "rainbow + infer"], title="rewards through training epoch" \
                        , save_destination=os.path.join(path_dict['picture'], "reward_curves_i{}.png".format(plot_id)),
                        args=args)

# os.system('rm -rf ./data/Store_datatype*.pickle')
