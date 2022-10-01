import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

result = []
group_dict = {
    1:'32x32 mlp',
    2:'32x32x32 mlp',
    3:'64x64 mlp',
    4:'64x64x64 mlp',
    5:'128x128 mlp',
    6:'128x128x128 mlp'
}
## load training data
for i in range(6):
    with open('./data/Store_datatype_{}.pickle'.format(i+1), 'rb') as f:
        files = pickle.load(f)
        data = list(map(lambda x: x.append(group_dict[i+1]),list(map(list, files))))
        result += data

df = pd.DataFrame(result, columns = ['epochs','MSE','network type'])
sns.set()
figure = plt.figure()
sns.lineplot(x="epochs",y="MSE",hue="network type",data=df)
plt.xlabel = "training epochs"
plt.ylabel = "Mean Squared Error"
plt.title = "capacity loss of neural network on MNIST"
fig.savefig("./Mse_Fig_MNIST.png")
