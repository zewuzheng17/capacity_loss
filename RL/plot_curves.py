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
    path_dict = create_path_dict(args)
    plot_fig(path = os.path.join(path_dict['data'],"curves_loss.pkl"), xlabel="epoch", ylabel="MSE", title="Mean Squared error through training epoch" \
             , save_destination=os.path.join(path_dict['picture'],"loss_curves.png"))
    plot_fig(path = os.path.join(path_dict['data'],"curves_EFdimension.pkl"), xlabel="epoch", ylabel="Effective dimension", title="Feature rank through training epoch" \
             , save_destination=os.path.join(path_dict['picture'],"Efdimension_curves.png"))

#os.system('rm -rf ./data/Store_datatype*.pickle')
