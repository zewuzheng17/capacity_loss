import os
import numpy as np
import pickle
from tqdm import tqdm
from functools import reduce
from algorithm.atari_network import Rainbow
from algorithm.atari_wrapper import make_atari_env
from util.utils import create_path_dict, smooth
from util.config_args import get_args

import ray
import torch
from torch.functional import F

from tianshou.data import PrioritizedVectorReplayBuffer


def setup_seed(seed):
    torch.manual_seed(seed)


@ray.remote(num_gpus=0.3)
def test_capacity(args, buffer) -> list[tuple]:
    total_loss = []
    seeds = np.random.randint(1000)
    setup_seed(seeds)
    # generate random network label
    net_random = Rainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        'cuda',
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy,
        add_infer=args.add_infer,
        infer_multi_head_num=args.infer_multi_head_num,
        infer_output_dim=args.infer_output_dim
    ).to('cuda')

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

    net_random.train(False)
    net.train(False)
    support = np.linspace(args.v_min, args.v_max, num = args.num_atoms)
    supports = torch.tensor(np.array([[support]])).detach().to('cuda')
    training_data, _ = buffer.sample(int(args.supervised_data_size)) # .sample(batch_size=args.supervised_data_size)
    training_data = training_data.obs

    with torch.no_grad():
        logit, _, _, _, _ = net_random(training_data)
        # logit = logit.to('cuda')
        label = torch.sum(logit * supports, dim=2)

    with tqdm(range(0, args.test_capacity_length)) as t:
        for i in t:
            # get checkpoint
            checkpoint = torch.load(
                os.path.join(path_dict['policy'],
                             "policy_i{}_s{}_{}.pth".format(args.add_infer, args.policy_save_seed, i + 1)))

            # load optimizer state and network parameters from checkpoint
            net.load_state_dict(checkpoint['model'])
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            optim.load_state_dict(checkpoint['optimizer'])

            # train for test capacity
            lossed = 0
            with torch.no_grad():
                logits_init, _, _, _, _ = net(training_data)
                output_init = torch.sum(logits_init * supports, dim=2)
                loss_init = torch.mean(torch.sum(torch.pow(output_init - label, 2), dim=1))
                loss_init = loss_init.item()
                print("Initial loss:", loss_init)

            for j in range(args.supervised_epoch):
                idx = np.random.choice(args.supervised_data_size - 1, size=args.test_batch_size, replace=True)
                batch = training_data[idx]
                labels = label[idx]
                optim.zero_grad()
                logits, _, _, _ , _= net(batch)
                output = torch.sum(logits * supports, dim=2)
                loss = torch.sum(torch.pow(output - labels, 2), dim = 1)
                loss = torch.mean(loss)

                loss.backward()
                torch.nn.utils.clip_grad_value_(net.parameters(), 40)
                optim.step()
                if j >= args.supervised_epoch - 10:
                    lossed += loss.item()
            t.set_postfix({"loss": lossed / 10})
            total_loss.append((i, lossed / 10, loss_init))
        # total_loss_smoothed = smooth(total_loss)
        return total_loss


def test_dimension(args, buffer) -> list[tuple]:
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
    net_r.train(False)
    for i in tqdm(range(args.test_capacity_length)):
        # get checkpoint
        checkpoint = torch.load(
            os.path.join(path_dict['policy'],
                         "policy_i{}_s{}_{}.pth".format(args.add_infer, args.policy_save_seed, i + 1)))


        representaion_data, _ = buffer.sample(batch_size=5000)
        representaion_data = representaion_data.obs
        net_r.load_state_dict(checkpoint['model'])
        optim = torch.optim.Adam(net_r.parameters(), lr=args.lr)
        optim.load_state_dict(checkpoint['optimizer'])

        # get effective dimension
        with torch.no_grad():
            _, _, net_represent, _ , _= net_r(representaion_data)
            singular_values = np.array(torch.linalg.svdvals(net_represent))
            effective_dimension = np.sum((singular_values) > 0.01)
            # print("max dim:", np.max(singular_values))
            total_effective_dimension.append((i, effective_dimension))
    return total_effective_dimension


if __name__ == "__main__":
    ray.init(num_cpus=10)
    args = get_args()
    args.add_infer = int(args.add_infer)
    args.algo_name = 'rainbow'
    path_dict = create_path_dict(args)
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

    buffer_total = PrioritizedVectorReplayBuffer.load_hdf5(os.path.join(path_dict['buffer'],
                                                                  "data_i{}_s{}.hdf5".format(args.add_infer,
                                                                                                args.policy_save_seed)))
    # #
    result_loss = ray.get([test_capacity.remote(args, buffer_total) for i in range(5)])
    total_result_loss = reduce(lambda x, y: x + y, result_loss)
    total_result_dimension = test_dimension(args, buffer_total)

    files = open(
        os.path.join(path_dict['data'], "curves_loss_i{}_s{}.pkl".format(args.add_infer, args.policy_save_seed)), 'wb')
    pickle.dump(total_result_loss, files)
    files.close()
    files = open(
        os.path.join(path_dict['data'], "curves_EFdimension_i{}_s{}.pkl".format(args.add_infer, args.policy_save_seed)),
        'wb')
    pickle.dump(total_result_dimension, files)
    files.close()


