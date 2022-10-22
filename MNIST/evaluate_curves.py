import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from parse_config import config_parser

parser = config_parser()
args = parser.parse_args()
result = []

if args.multi_network:
    group_dict = {
        1:"32x32 mlp",
        2:"32x32x32 mlp",
        3:"64x64 mlp",
        4:"64x64x64 mlp",
        5:"128x128 mlp",
        6:"128x128x128 mlp"
    }
else:
    group_dict = {
        1:"1",
        10:"10",
        100:"100"
    }
## load training data
for i in group_dict.keys():
    with open('./data/Store_datatype_i{}_{}.pickle'.format(args.add_infer, i), 'rb') as f:
        files = pickle.load(f)
        data = list(map(list, files))
        [x.append(group_dict[i]) for x in data]
        result += data

with open('./data/Store_datatype_i{}_{}.pickle'.format("False", 10), 'rb') as f:
    files = pickle.load(f)
    data = list(map(list, files))
    [x.append("No infer") for x in data]
    result += data

def plot_fig(data, x, y, hue, args):
    fig = plt.figure()
    sns.lineplot(x=x, y=y, hue=hue, data=data)
    plt.xlabel = x
    plt.ylabel = y
    plt.title = y
    plt.legend(loc="upper left", fontsize="x-small")
    fig.savefig("./picture/{}_Fig_MNIST_{}.png".format(y, int(args.add_infer)))

sns.set()

if args.multi_network:
    fig = plt.figure()
    df = pd.DataFrame(result, columns = ['epochs','MSE','network type'])
    plot_fig(data=df, x="epochs", y='MSE', hue='network type', args=args)
else:
    df = pd.DataFrame(result, columns = ['epochs','MSE','representation_mean', 'ef_dimension', 'scale_test'])
    plot_fig(data=df, x="epochs", y='MSE', hue='scale_test', args=args)
    plot_fig(data=df, x="epochs", y='representation_mean', hue='scale_test', args=args)
    plot_fig(data=df, x="epochs", y='ef_dimension', hue='scale_test', args=args)



