import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import random_split
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
        self.fc = nn.Sequential(*fc_h)

        self.fc_out = nn.Linear(self._size[self._hidden_layers - 1], self._size[self._hidden_layers])

        if self.args.add_infer:
            self.infer_out = nn.Linear(self._size[self._hidden_layers - 1], 10)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.fc(x) # scalar output
        fc_out = self.fc_out(x)
        if self.args.add_infer:
            infer_out = self.infer_out(x)
        else:
            infer_out = 0
        return fc_out, infer_out, x

class Random_Net(nn.Module):
    def __init__(self, input_dim):
        super(Random_Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30,1)
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x) # scalar output
        return x

@ray.remote(num_gpus=0.3)
class MLP_Train():
    def __init__(self, model, args) -> None:
        self.model = model
        self.model_infer = copy.deepcopy(model).to('cuda')
        self.args = args
        self.optim = torch.optim.Adam(model.parameters(), lr = self.args.learning_rate) 
        self.loss = nn.MSELoss()    
        self.training_loss = []

    def train_random_target(self, data, random_net):
        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        try:
            with tqdm(range(self.args.total_epoch)) as t:
                for k in t:
                    j = 0
                    total_epoch_loss = 0
                    seed = np.random.randint(1000)
                    setup_seed(seed)

                    for i in range(self.args.num_epochs):
                        for image, label in iter(data):
                            image = image.view(-1, 784).to('cuda')
                            with torch.no_grad():
                                label = random_net[k].forward(image)
                                label = torch.sin(label * 1000)
                                if self.args.add_infer:
                                    _, infer_label, _ = self.model_infer(image)
                            self.optim.zero_grad()
                            output, infer_output, representation = self.model.forward(image)
                            mean_representation = torch.mean(representation.detach()).item()
                            singular_values = np.array(torch.linalg.svdvals(representation.detach().to('cpu')))
                            effective_dimension = np.sum((singular_values) > 0.01)
                            origin_loss = self.loss(output, label)
                            if self.args.add_infer:
                                infer_loss = 0.1 * self.loss(infer_output, infer_label * 1)
                                loss = origin_loss + infer_loss
                            else:
                                loss = origin_loss
                            loss.backward()
                            self.optim.step()
                            if i == self.args.num_epochs - 1:
                                if self.args.add_infer:
                                    total_epoch_loss += origin_loss.item()
                                else:
                                    total_epoch_loss += loss.item()
                                j += 1
                    t.set_description("step {}".format(k))
                    if not self.args.add_infer:
                        t.set_postfix({"loss":total_epoch_loss/j})
                    else:
                        t.set_postfix({"origin loss":origin_loss.item(), "infer_loss":infer_loss.item()})
                    self.training_loss.append((k,total_epoch_loss/j, mean_representation, effective_dimension))
                    #writer.add_scalar("loss\origin loss", origin_loss.item(), i)
                    #writer.add_scalar("loss\infer loss", infer_loss.item(), i)
                    #j += 1
                return self.training_loss
        except KeyboardInterrupt:
            t.close()
            raise Exception("Being Interarupted by keyboard")


class MLP_Train_test():
    def __init__(self, model, args) -> None:
        self.model = model
        self.model_infer = copy.deepcopy(model).to('cuda')
        self.args = args
        self.optim = torch.optim.Adam(model.parameters(), lr = self.args.learning_rate)
        self.loss = nn.MSELoss()
        self.training_loss = []

    def train_random_target(self, data, random_net, writer):
        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        try:
            with tqdm(range(self.args.total_epoch)) as t:
                for k in t:
                    j = 0
                    total_epoch_loss = 0
                    seed = np.random.randint(1000)
                    setup_seed(seed)

                    for i in range(self.args.num_epochs):
                        for image, label in iter(data):
                            image = image.view(-1, 784).to('cuda')
                            with torch.no_grad():
                                label = random_net[k].forward(image)
                                label = torch.sin(label * 1000)
                                if self.args.add_infer:
                                    _, infer_label = self.model_infer(image)
                            self.optim.zero_grad()
                            output, infer_output = self.model.forward(image)
                            origin_loss = self.loss(output, label)
                            if self.args.add_infer:
                                infer_loss = 0.1 * self.loss(infer_output, infer_label * 10)
                                loss = origin_loss + infer_loss
                            else:
                                loss = origin_loss
                            loss.backward()
                            self.optim.step()
                            if i == self.args.num_epochs - 1:
                                total_epoch_loss += loss.item()
                                j += 1
                    t.set_description("step {}".format(k))
                    t.set_postfix({"loss": total_epoch_loss / j})
                    writer.add_scalar("origin loss", origin_loss.item(), k)
                    self.training_loss.append((k ,total_epoch_loss/j))
                        #writer.add_scalar("loss\infer loss", infer_loss.item(), i)
                    #j += 1
                return self.training_loss
        except KeyboardInterrupt:
            t.close()
            raise Exception("Being Interarupted by keyboard")

if __name__ == "__main__":
    ## argments
    ray.init(num_cpus=6)
    parser = config_parser()
    args = parser.parse_args()

    ## data set loading
    normalize = transforms.Normalize(mean=[.5], std=[.5])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = torchvision.datasets.MNIST(root='./data/', train = True, transform = transform, download = False)
    datas, _ = random_split(dataset, lengths=(args.num_samples,59000))
    train_loader = data.DataLoader(datas, batch_size = args.batch_size, drop_last=True)
    summary = SummaryWriter("tensorlog/{}".format(args.pickle_type))

    random_net = [Random_Net(input_dim=784).to('cuda') for i in range(args.total_epoch)]
    mlp_networks = [MLP_Network(input_dim = 784, args = args).to('cuda') for i in range(args.num_seeds)]
    pool = [MLP_Train.remote(mlp_networks[i], args) for i in range(args.num_seeds)]
    future = [t.train_random_target.remote(train_loader, random_net) for t in pool]
    results = ray.get(future)

    # mlp_networks = MLP_Network(input_dim = 784, args = args).to('cuda')
    # train = MLP_Train_test(mlp_networks, args)
    # results = train.train_random_target(train_loader, random_net, writer=summary)

    total_results = reduce(lambda x, y: x + y, results)

    ## save result for plots
    # if args.multi_network:
    #     files = open('./data/Store_datatype_i{}_{}.pickle'.format(args.add_infer, args.pickle_type), 'wb')
    # else:
    #     files = open('./data/Store_datatype_i{}_{}.pickle'.format(args.add_infer, args.num_epochs), 'wb')
    files = open('./data/Store_datatype_i{}_1.pickle'.format(args.add_infer), 'wb')
    pickle.dump(total_results, files)
    files.close()
    summary.close()
    # ray.shutdown()