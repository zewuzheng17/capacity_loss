# __package__ = 'RL.evaluate'
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from util.config_args import get_args
from util.utils import create_path_dict

def plot_fig(path, xlabel, ylabel, title, save_destination):
## load training data
    with open(path, 'rb') as f:
        files = pickle.load(f)
        data = list(map(list, files))

    df = pd.DataFrame(data, columns=[xlabel, ylabel])
    sns.set()
    fig = plt.figure()
    sns.lineplot(x=xlabel, y=ylabel, data=df)  # ,hue="network type"
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.title = title
    plt.legend(loc="upper left", fontsize="x-small")
    fig.savefig(save_destination)

if __name__ == "__main__":
    args = get_args()
    args.add_infer = int(args.add_infer)
    path_dict = create_path_dict(args)
    plot_fig(path = os.path.join(path_dict['data'],"curves_loss_{}.pkl".format(args.add_infer)), xlabel="epoch", ylabel="MSE", title="Mean Squared error through training epoch" \
             , save_destination=os.path.join(path_dict['picture'],"loss_curves_{}.png".format(args.add_infer)))
    plot_fig(path = os.path.join(path_dict['data'],"curves_EFdimension_{}.pkl".format(args.add_infer)), xlabel="epoch", ylabel="Effective dimension", title="Feature rank through training epoch" \
             , save_destination=os.path.join(path_dict['picture'],"Efdimension_curves_{}.png".format(args.add_infer)))
    plot_fig(path = os.path.join(path_dict['data'],"rewards_{}.pkl".format(args.add_infer)), xlabel="epoch", ylabel="Rewards", title="rewards through training epoch" \
             , save_destination=os.path.join(path_dict['picture'],"Rewards_curves_{}.png".format(args.add_infer)))

#os.system('rm -rf ./data/Store_datatype*.pickle')
