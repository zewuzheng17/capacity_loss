# __package__ = 'RL.algorithm'

import os
import pprint
import numpy as np
import torch

from util.utils import create_path_dict
from util.config_args import get_args
from algorithm.atari_network import Rainbow
from algorithm.atari_wrapper import make_atari_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.policy import RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger



def test_rainbow(args=get_args()):
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
    # should be N_FRAMES x H x W
    print("environment:", args.task)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("device", args.device)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model, basically, this is the network structure
    net = Rainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        args.device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy
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
        target_update_freq=args.target_update_freq
    ).to(args.device)
    # load a previous policy, if training is suspended, we can resume the policy here
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
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
    # collector, which collect data using define policy, and store the data into replay buffer
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)


    # logger
    path_dict = create_path_dict(args)
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=path_dict['base'].replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(path_dict['tensor_log'])
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    # save the best policy into policy.pth using torch.save
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(path_dict['policy'], "policy.pth"))

    def save_checkpoint_fn(policy, epochs, buffer, step_per_epoch, save_interval=args.checkpoint_save_interval):
        assert save_interval >= step_per_epoch  # ensure that we minimally save at least each epoch

        if epochs == 1:
            buffer.save_hdf5(os.path.join(path_dict['buffer'],
                                               "data0.hdf5"))
            print("The first replay buffer saved")
        if epochs % int(save_interval / step_per_epoch) == 0:
            checkpoint = {
                "model": policy.model.state_dict(),
                "optimizer": policy.optim.state_dict()
            }
            torch.save(checkpoint, os.path.join(path_dict['policy'], "policy_{}M.pth".format(
                int(epochs / int(save_interval / step_per_epoch)))))

    # the stopping criteria of training
    def stop_fn(mean_rewards):
        # if env.spec.reward_threshold:
        #     return mean_rewards >= env.spec.reward_threshold
        # elif "Pong" in args.task:
        #     return mean_rewards >= 20
        # else:
        return False

    # set the learning rate, prioritize experience replay rate alpha and beta..
    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay the exploration rate in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                  (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * \
                       (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        # set to evaluation mode
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
                alpha=args.alpha,
                beta=args.beta
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    test_rainbow(get_args())
