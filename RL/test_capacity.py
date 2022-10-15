
import os
import numpy as np
import pickle
from tqdm import tqdm
from functools import reduce

from algorithm.atari_network import Rainbow
from algorithm.atari_wrapper import make_atari_env
from util.utils import create_path_dict
from util.config_args import get_args

import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.functional import F
from torch.utils.data import RandomSampler, BatchSampler

from tianshou.data import PrioritizedVectorReplayBuffer


# def copy_from_policy_dict(source):
#     new_dict = {}
#     patt = re.compile(r'model_old')
#     for keys in source.keys():
#         if patt.findall(keys) == []:
#             new_dict[keys.lstrip('model.')] = source[keys]
#     del new_dict['support']
#     return new_dict


@ray.remote(num_gpus=0.3,num_cpus=2)
def test_capacity(args) -> list[tuple]:
    # get buffer
    buffer = PrioritizedVectorReplayBuffer.load_hdf5(os.path.join(path_dict['buffer'], "data_i{}_s{}.hdf5".format(args.add_infer, args.seed)))
    # get checkpoint
    checkpoint = [torch.load(os.path.join(path_dict['policy'], "policy_i{}_s{}_{}.pth".format( args.add_infer, args.seed,i + 1))) for i in range(args.test_capacity_length)]

    training_data, _ = buffer.sample(batch_size=args.supervised_data_size)
    training_data = training_data.obs
    # generate random network label
    net_random = Rainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        args.device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy,
        add_infer=args.add_infer,
        infer_multi_head_num=args.infer_multi_head_num,
        infer_output_dim=args.infer_output_dim
    ).to(args.device)

    label, _, _, _ = net_random(training_data)
    label = label.detach() * 10

    sampler_idx = list(
        BatchSampler(RandomSampler(range(args.supervised_data_size)), batch_size=args.batch_size, drop_last=True))

    total_loss = []
    net = Rainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        args.device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy,
        add_infer=args.add_infer,
        infer_multi_head_num=args.infer_multi_head_num,
        infer_output_dim=args.infer_output_dim
    ).to(args.device)

    for i in tqdm(range(args.test_capacity_length)): #

        # load optimizer state and network parameters from checkpoint
        net.load_state_dict(checkpoint[i]['model'])
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        optim.load_state_dict(checkpoint[i]['optimizer'])

        # train for test capacity
        loss_epoch = 0
        for j in range(args.supervised_epoch):
            for idx in sampler_idx:
                batch = training_data[idx]
                labels = label[idx]
                optim.zero_grad()

                logits, _, _, _ = net(batch)
                loss = F.mse_loss(logits, labels)
                loss.backward()
                optim.step()
            # if writer is not None:
            #     writer.add_scalar("MSE", j, loss.item())
        total_loss.append((i, loss.item()))
    return total_loss

def test_dimension(args) -> list[tuple]:
    # get buffer
    buffer = PrioritizedVectorReplayBuffer.load_hdf5(os.path.join(path_dict['buffer'], "data_i{}_s{}.hdf5".format(args.add_infer, args.seed)))
    # get checkpoint
    checkpoint = [torch.load(os.path.join(path_dict['policy'], "policy_i{}_s{}_{}.pth".format(args.seed, args.add_infer, i + 1))) for i in range(args.test_capacity_length)]

    representaion_data, _ = buffer.sample(batch_size=50000)
    representaion_data = representaion_data.obs

    total_effective_dimension = []
    # cpu net for computing large batch size
    net_r = Rainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        'cpu',
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy,
        add_infer=args.add_infer,
        infer_multi_head_num=args.infer_multi_head_num,
        infer_output_dim=args.infer_output_dim
    )

    for i in tqdm(range(args.test_capacity_length)):
        net_r.load_state_dict(checkpoint[i]['model'])
        optim = torch.optim.Adam(net_r.parameters(), lr=args.lr)
        optim.load_state_dict(checkpoint[i]['optimizer'])

        # get effective dimension
        with torch.no_grad():
            _, _, net_represent, _ = net_r(representaion_data)
            effective_dimension = np.sum(np.array(torch.linalg.svdvals(net_represent)) > 0.01)
            total_effective_dimension.append((i, effective_dimension))
    return total_effective_dimension


if __name__ == "__main__":
    args = get_args()
    args.add_infer = int(args.add_infer)
    path_dict = create_path_dict(args)
    writer = SummaryWriter(path_dict['tensor_log'])

    ## get env action_shape and state_shape
    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    result_loss= ray.get([test_capacity.remote(args) for i in range(5)])
    total_result_loss = reduce(lambda x, y: x + y, result_loss)
    total_result_dimension = test_dimension(args)

    files = open(os.path.join(path_dict['data'],"curves_loss_i{}_s{}.pkl".format(args.add_infer, args.seed)), 'wb')
    pickle.dump(total_result_loss, files)
    files.close()
    files = open(os.path.join(path_dict['data'],"curves_EFdimension_i{}_s{}.pkl".format(args.add_infer, args.seed)), 'wb')
    pickle.dump(total_result_dimension, files)
    files.close()
    writer.close()
