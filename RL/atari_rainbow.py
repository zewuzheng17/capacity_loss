# __package__ = 'RL.algorithm'

import os
import pprint
import numpy as np
import torch
import pickle

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
    # set working numbers of gpu and seeds
    torch.set_num_threads(int(args.training_num) + 1)
    args.seed = np.random.randint(100)
    print("seeds:", args.seed)

    if args.add_infer:
        print("add infer for training")
    else:
        print("training without infer")
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
        gradnorm = args.grad_norm,
        target_update_freq=args.target_update_freq,
        add_infer=args.add_infer,
        infer_gradient_scale=args.infer_gradient_scale,
        infer_target_scale=args.infer_target_scale,
        global_grad_norm = args.global_grad_norm
    ).to(args.device)

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

    path_dict = create_path_dict(args)

    if args.resume:
        buffer_path = os.path.join(path_dict['buffer'], "data_recent_i{}_s{}.hdf5".format(args.add_infer, args.policy_save_seed))
        if os.path.exists(buffer_path):
            buffer = buffer.load_hdf5(buffer_path)
            print("Successfully restore buffer from {}.".format("data_recent_i{}_s{}.hdf5".format(args.add_infer, args.policy_save_seed)))
        else:
            raise Exception("Fail to restore buffer:{} ,pls check if file exist".format(buffer_path))

        for i in range(100, 0, -1):
            if os.path.exists(
                    os.path.join(path_dict['policy'], "policy_i{}_s{}_{}.pth".format(args.add_infer, args.policy_save_seed, i))):
                checkpoint_path = os.path.join(path_dict['policy'], "policy_i{}_s{}_{}.pth".format(args.add_infer, args.policy_save_seed, i))
                checkpoint = torch.load(checkpoint_path)
                policy.model.load_state_dict(checkpoint['model'])
                policy.optim.load_state_dict(checkpoint['optimizer'])
                print("Successfully restore policy and optim.")
                break

            if i == 1:
                raise Exception("Fail to load checkpoint")

    # collector, which collect data using define policy, and store the data into replay buffer
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True) if args.collect_test_statistics else None

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=path_dict['base'].replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    # tensor_log
    ts_log_path = os.path.join(path_dict['tensor_log'],str(args.add_infer) + str(args.policy_save_seed))
    if not os.path.exists(ts_log_path):
        os.makedirs(ts_log_path)
    writer = SummaryWriter(ts_log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    # save the best policy into policy.pth using torch.save
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(path_dict['policy'], "policy.pth"))

    def save_checkpoint_fn(policy, epochs, buffer, step_per_epoch, test_reward: float,
                           save_interval=args.checkpoint_save_interval):
        assert save_interval >= step_per_epoch  # ensure that we minimally save at least each epoch

        if epochs == 1:
            buffer.save_hdf5(os.path.join(path_dict['buffer'],
                                          "data_i{}_s{}.hdf5".format(args.add_infer, args.policy_save_seed)))
            with open(os.path.join(path_dict['data'], "rewards_i{}_s{}.pkl".format(args.add_infer, args.policy_save_seed)), "wb") as f:
                pickle.dump([(epochs, test_reward)], f)
            print("The first replay buffer and rewards stats saved")
        elif epochs % int(save_interval / step_per_epoch) == 0:
            with open(os.path.join(path_dict['data'], "rewards_i{}_s{}.pkl".format(args.add_infer,args.policy_save_seed)), "rb") as f:
                reward_list = pickle.load(f)

            reward_list += [(epochs, test_reward)]
            with open(os.path.join(path_dict['data'], "rewards_i{}_s{}.pkl".format(args.add_infer,args.policy_save_seed)), "wb") as f:
                pickle.dump(reward_list, f)

            checkpoint = {
                "model": policy.model.state_dict(),
                "optimizer": policy.optim.state_dict()
            }
            torch.save(checkpoint, os.path.join(path_dict['policy'], "policy_i{}_s{}_{}.pth".format(
                args.add_infer, args.policy_save_seed, int(epochs / int(save_interval / step_per_epoch)) )))

            # buffer.save_hdf5(os.path.join(path_dict['buffer'],
            #                               "data_i{}_s{}_{}.hdf5".format(args.add_infer, args.policy_save_seed, \
            #                                                             int(epochs / int(save_interval / step_per_epoch))))\
            #                  , compression=True)


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
    #watch()


if __name__ == "__main__":
    test_rainbow(get_args())
