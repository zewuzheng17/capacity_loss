import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from tqdm import tqdm
from util.config_args_mujoco import get_args
from util.utils import create_path_dict
from algorithm.mujoco_env import make_mujoco_env
from tianshou.data import ReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


def test_stats(actor, critic, path_dict):
    srankp, srankq, efp, efq = [], [], [], []
    for reset in range(2):
        for seeds in range(6):
            buffer_path = os.path.join(path_dict['buffer'], 'data_i{}_s{}.hdf5'.format(int(reset),
                                                                                       int(seeds)))
            buffer = ReplayBuffer.load_hdf5(buffer_path)
            batch, idx = buffer.sample(50000)

            obs = torch.as_tensor(
                batch.obs,
                device=critic1.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)

            act = torch.as_tensor(
                batch.act,
                device=critic1.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)
            obsd = torch.cat([obs, act], dim=1)

            for i in tqdm(range(1, 200)):
                policy_path = os.path.join(path_dict['policy'], 'policy_r{}_s{}_{}.pth'.format(int(reset),
                                                                                               int(seeds), i))
                checkpoint = torch.load(policy_path)
                actor.load_state_dict(checkpoint['actor'])
                critic.load_state_dict(checkpoint['critic1'])

                with torch.no_grad():
                    net_represent_q, hidden_q = critic1.preprocess(obsd)
                    net_represent_p, hidden_p = actor.preprocess(obs, state=None)
                    singular_values_q = np.array(torch.linalg.svdvals(net_represent_q).cpu())
                    singular_values_p = np.array(torch.linalg.svdvals(net_represent_p).cpu())
                    # get effective dimension of actor and critic
                    effective_dimension_q = np.sum((singular_values_q / np.max(singular_values_q)) > 0.01)
                    effective_dimension_p = np.sum((singular_values_p / np.max(singular_values_p)) > 0.01)

                    # get srank for actor and critic
                    srank_q_idx = np.array(
                        [(np.sum(singular_values_q[:i + 1]) / np.sum(singular_values_q)) > 0.99 for i in
                         range(len(singular_values_q))])

                    srank_q = np.where(srank_q_idx == 1)[0][0]
                    srank_p_idx = np.array(
                        [(np.sum(singular_values_p[:i + 1]) / np.sum(singular_values_p)) > 0.99 for i in
                         range(len(singular_values_p))])
                    srank_p = np.where(srank_p_idx == 1)[0][0]
                    if seeds / 3 < 1 and reset == 0:
                        loc = "SAC, RR=1"
                    elif reset == 0:
                        loc = "SAC, RR=9"
                    elif seeds / 3 < 1:
                        loc = "SAC + reset, RR=1"
                    else:
                        loc = "SAC + reset, RR=9"

                    srankp.append((i, srank_p, loc))
                    srankq.append((i, srank_q, loc))
                    efp.append((i, effective_dimension_p, loc))
                    efq.append((i, effective_dimension_q, loc))
    return srankp, srankq, efp, efq


def plot_fig_single(data, xlabel, ylabel, title, save_destination):
    dash_dict = {
        "SAC, RR=1": (2, 2),
        "SAC, RR=9": (2, 2),
        "SAC + reset, RR=1": (1, 0),
        "SAC + reset, RR=9": (1, 0)
    }

    color = ['blue', 'green', 'orange', 'red']
    sns.set()
    sns.set_palette(sns.color_palette(color))
    fig = plt.figure()
    data_l = list(map(lambda x: list(x), data))
    df = pd.DataFrame(data_l, columns=[xlabel, ylabel, "type"])

    sns.lineplot(x=xlabel, y=ylabel, data=df, hue="type", style="type", dashes=dash_dict)
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize='small', ncol = 2)
    plt.tight_layout()
    fig.savefig(save_destination)


if __name__ == "__main__":
    args = get_args()
    path_dict = create_path_dict(args)
    torch.set_num_threads(10)
    env, train_envs, test_envs = make_mujoco_env(
        args.task, args.seed, args.training_num, args.test_num, obs_norm=False, dmc_control=args.dmc
    )

    if args.dmc:
        args.state_shape = env.observation_spaces.shape or env.observation_spaces.n
    else:
        args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)

    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    # critics return a single q-value of state-action pairs
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    srankp, srankq, efp, efq = test_stats(actor, critic1, path_dict)
    plot_fig_single(srankp, "epochs", "srank actor", "srank for {}".format(args.task),
                    os.path.join(path_dict['picture'],
                                 '{}_srank_p.png'.format(args.task)))
    plot_fig_single(srankq, "epochs", "srank critic", "srank for {}".format(args.task),
                    os.path.join(path_dict['picture'],
                                 '{}_srank_q.png'.format(args.task)))
    plot_fig_single(efp, "epochs", "effective dimension actor", "effective dimension for {}".format(args.task),
                    os.path.join(path_dict['picture'],
                                 '{}_ef_p.png'.format(args.task)))
    plot_fig_single(efq, "epochs", "effective dimension critic", "effective dimension for {}".format(args.task),
                    os.path.join(path_dict['picture'],
                                 '{}_ef_q.png'.format(args.task)))

    # save stats
    with open(os.path.join(path_dict['data'], '{}_srank_p.pkl'.format(args.task)), 'wb') as f:
        pickle.dump(srankp, f)
    with open(os.path.join(path_dict['data'], '{}_srank_q.pkl'.format(args.task)), 'wb') as f:
        pickle.dump(srankq, f)
    with open(os.path.join(path_dict['data'], '{}_ef_p.pkl'.format(args.task)), 'wb') as f:
        pickle.dump(efp, f)
    with open(os.path.join(path_dict['data'], '{}_ef_q.pkl'.format(args.task)), 'wb') as f:
        pickle.dump(efq, f)
