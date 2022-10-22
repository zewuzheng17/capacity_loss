import os
import numpy as np
import torch

from util.utils import create_path_dict
from util.config_args import get_args
from algorithm.atari_network import Rainbow
from algorithm.atari_wrapper import make_atari_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.policy import RainbowPolicy

args = get_args()
args.add_infer = int(args.add_infer)
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
)
optim = torch.optim.Adam(net.parameters(), lr=args.lr)
# define policy, this is to define optimizer, the logit of computing q value, including noisy network,
# distributional q, deal with returns, including GAE, n-step return, and define the loss function~, get action.... etc
policy = RainbowPolicy(
    net,
    optim,
    args.gamma,
    args.num_atoms,
    args.v_min,
    args.v_max,
    args.n_step,
    gradnorm=args.grad_norm,
    target_update_freq=args.target_update_freq,
    add_infer=args.add_infer,
    infer_gradient_scale=args.infer_gradient_scale,
    infer_target_scale=args.infer_target_scale,
    global_grad_norm=args.global_grad_norm
).to(args.device)

if args.no_priority:
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack
    )
else:
    buffer = PrioritizedVectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack,
        alpha=args.alpha,
        beta=args.beta,
        weight_norm=not args.no_weight_norm
    )

path_dict = create_path_dict(args)
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True) if args.collect_test_statistics else None

class Policys:
    def __init__(self, value):
        self.value = value

p = Policys(99)

class Test_change_policy:
    def __init__(self, policy):
        self.policy = policy

    def change(self):
        self.policy.value = 12345678

class collect:
    def __init__(self, policy):
        self.policy = policy

test_collect = collect(p)
print(test_collect.policy.value)
tpolicy = Test_change_policy(p)
tpolicy.change()
print(test_collect.policy.value)