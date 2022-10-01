import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from functools import reduce
from parse_config import config_parser
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ray
import pickle


class MLP_Network(nn.Module):
    def __init__(self, input_dim, args):
        super(MLP_Network, self).__init__()
        self.args = args
        self._size = [input_dim] + list(map(int, self.args.hidden_size.split(' ')))
        self._hidden_layers = len(self._size) - 1
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][self.args.activation_id]

        fc_h = []
        for j in range(self._hidden_layers):
            if j != self._hidden_layers - 1:
                fc_h += [
                    nn.Linear(self._size[j], self._size[j + 1]), active_func, nn.LayerNorm(self._size[j + 1])
                ] # 
            else:
                fc_h += [
                    nn.Linear(self._size[j], self._size[j + 1])
                ]
        self.fc = nn.Sequential(*fc_h)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = self.softmax(x)
        return x

@ray.remote(num_cpus=2)
class MLP_Train():
    def __init__(self, model, args) -> None:
        self.model = model
        self.args = args
        self.optim = torch.optim.Adam(model.parameters(), lr = self.args.learning_rate) 
        self.loss = nn.MSELoss()    
        self.training_loss = []

    def train(self, data, writer):
        j = 0
        for i in tqdm(range(self.args.num_epochs)):
            for image, label in iter(data):
                image = image.view(-1, 784)
                label = F.one_hot(label).to(torch.float32)
                self.optim.zero_grad()
                
                output = self.model.forward(image)
                loss = self.loss(output, label)
                loss.backward()
                self.optim.step()
            self.training_loss.append(loss.item())
            writer.add_scalar("MSE", loss.item(), j)
            j += 1

    def train_random_target(self, data):
        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            #np.random.seed(seed)
            #random.seed(seed)
            torch.backends.cudnn.deterministic = True

        try:
            with tqdm(range(self.args.total_epoch)) as t:
                for k in t:
                    j = 0
                    total_epoch_loss = 0
                    seed = np.random.randint(1000)
                    setup_seed(seed)
                    random_label_generator = MLP_Network(input_dim = 784, args = self.args)
                    for i in range(self.args.num_epochs):
                        for image, label in iter(data):
                            image = image.view(-1, 784)
                            with torch.no_grad():
                                label = random_label_generator.forward(image)
                                label = self.softmax_to_onehot(label)
                            self.optim.zero_grad()
                            output = self.model.forward(image)
                            loss = self.loss(output, label)
                            loss.backward()
                            self.optim.step()
                            if i == self.args.num_epochs - 1:
                                total_epoch_loss += loss.item()
                                j += 1
                    self.training_loss.append((k ,total_epoch_loss/j))
                    #writer.add_scalar("MSE", loss.item(), j)
                    #j += 1
                return self.training_loss
        except KeyboardInterrupt:
            t.close()
            raise Exception("Being Interarupted by keyboard")
        t.close()


    def softmax_to_onehot(self, source):
        index = torch.argmax(source, dim = 1)
        target = torch.zeros_like(source)
        for row in range(target.size(0)):
            target[row][index[row]] = 1
        return target



if __name__ == "__main__":
    ## argments
    parser = config_parser()
    args = parser.parse_args()

    ## data set loading
    normalize = transforms.Normalize(mean=[.5], std=[.5])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = torchvision.datasets.MNIST(root='./data/', train = True, transform = transform, download = False)
    sampler = RandomSampler(dataset, num_samples=args.num_samples, replacement = True)
    train_loader = data.DataLoader(dataset, batch_size = args.batch_size, drop_last=True, sampler = sampler)
    #summary = SummaryWriter("~/tensorlog")

    ## Neural Network and Training
    # mlp_networks = MLP_Network(input_dim = 784, args)
    # Train = MLP_Train(mlp_networks, args)
    # Train.train(train_loader, summary)

    mlp_networks = [MLP_Network(input_dim = 784, args = args) for i in range(args.num_seeds)]
    pool = [MLP_Train.remote(mlp_networks[i], args) for i in range(args.num_seeds)]
    future = [t.train_random_target.remote(train_loader) for t in pool]
    results = ray.get(future)
    total_results = reduce(lambda x, y: x + y, results)

    ## save result for plots
    files = open('./data/Store_datatype_{}.pickle'.format(args.pickle_type), 'wb')
    pickle.dump(total_results, files)
    files.close()
    #summary.close()
    ray.shutdown()