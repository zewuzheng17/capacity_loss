import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

result = []
group_dict = {
    1:"32x32 mlp",
    2:"32x32x32 mlp",
    3:"64x64 mlp",
    4:"64x64x64 mlp",
    5:"128x128 mlp",
    6:"128x128x128 mlp"
}
## load training data
for i in range(6):
    with open('./data/Store_datatype_{}.pickle'.format(i+1), 'rb') as f:
        files = pickle.load(f)
        data = list(map(list, files))
        [x.append(group_dict[i+1]) for x in data]
        result += data

df = pd.DataFrame(result, columns = ['epochs','MSE','network type'])
sns.set()
fig = plt.figure()
sns.lineplot(x="epochs",y="MSE",hue="network type",data=df)
# plt.xlim(0,30)
# plt.ylim(0,0.4)
plt.xlabel = "training epochs"
plt.ylabel = "Mean Squared Error"
plt.title = "capacity loss of neural network on MNIST"
plt.legend(loc="upper left", fontsize="x-small")
fig.savefig("./picture/Mse_Fig_MNIST_recent.png")

os.system('rm -rf ./data/Store_datatype*.pickle')