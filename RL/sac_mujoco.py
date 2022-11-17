import datetime
import os
import pprint
import pickle

import numpy as np
import torch
from util.config_args_mujoco import get_args
from util.utils import create_path_dict
from algorithm.mujoco_env import make_mujoco_env
from torch.utils.tensorboard import SummaryWriter
# from algorithm.mujoco_env import WarpObs

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


def test_sac(args=get_args()):
    # set random seeds
    torch.set_num_threads(15)
    args.seed = np.random.randint(100)
    print("seeds:", args.seed)
    print("device:", args.device)
    env, train_envs, test_envs = make_mujoco_env(
        args.task, args.seed, args.training_num, args.test_num, obs_norm=False, dmc_control=args.dmc
    )
    # env, train_envs, test_envs = WarpObs(env), WarpObs(train_envs), WarpObs(test_envs)
    print('task:', args.task)
    print('reset interval:', args.reset_interval)
    if args.dmc:
        args.state_shape = env.observation_spaces.shape or env.observation_spaces.n
    else:
        args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    # base net for actor, input are states
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    # base net for critic, input are state_action
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    # critics return a single q-value of state-action pairs
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
        reset=args.reset,
        reset_interval=args.reset_interval,
        test_capacity=args.test_capacity,
        test_capacity_interval=args.test_capacity_interval
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    args.algo_name = 'sac'
    path_dict = create_path_dict(args)

    if args.resume:
        # buffer_path = os.path.join(path_dict['buffer'],
        #                            "data_recent_i{}_s{}.hdf5".format(args.reset, args.policy_save_seed))
        # if os.path.exists(buffer_path):
        #     buffer.load_hdf5(buffer_path)
        #     print("Successfully restore buffer from {}.".format(
        #         "data_recent_i{}_s{}.hdf5".format(args.reset, args.policy_save_seed)))
        # else:
        #     raise Exception("Fail to restore buffer:{} ,pls check if file exist".format(buffer_path))
        #
        # for i in range(100, 0, -1):
        #     if os.path.exists(
        #             os.path.join(path_dict['policy'],
        #                          "policy_i{}_s{}_{}.pth".format(args.reset, args.policy_save_seed, i))):
        #         checkpoint_path = os.path.join(path_dict['policy'],
        #                                        "policy_i{}_s{}_{}.pth".format(args.reset, args.policy_save_seed, i))
        #         checkpoint = torch.load(checkpoint_path)
        #         policy.model.load_state_dict(checkpoint['model'])
        #         policy.optim.load_state_dict(checkpoint['optimizer'])
        #         print("Successfully restore policy and optim.")
        #         break
        #
        #     if i == 1:
        #         raise Exception("Fail to load checkpoint")
        pass

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=path_dict['base'].replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )

    ts_log_path = os.path.join(path_dict['tensor_log'], str(args.reset) + str(args.policy_save_seed))
    if not os.path.exists(ts_log_path):
        os.makedirs(ts_log_path)
    writer = SummaryWriter(ts_log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(path_dict['policy'], "policy.pth"))

    def save_checkpoint_fn(policy, epochs, buffer, step_per_epoch, logger,
                           save_interval=args.checkpoint_save_interval):
        assert save_interval >= step_per_epoch  # ensure that we minimally save at least each epoch

        if epochs % int(save_interval / step_per_epoch) == 0:
            batch, indices = buffer.sample(5000)
            obs = torch.as_tensor(
                batch.obs,
                device=policy.critic1.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)

            act = torch.as_tensor(
                batch.act,
                device=policy.critic1.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)
            obsd = torch.cat([obs, act], dim=1)
            # compute effective dimension for representation
            with torch.no_grad():
                net_represent_q, hidden_q = policy.critic1.preprocess(obsd)
                net_represent_p, hidden_p = policy.actor.preprocess(obs, state=None)
                singular_values_q = np.array(torch.linalg.svdvals(net_represent_q).cpu())
                singular_values_p = np.array(torch.linalg.svdvals(net_represent_p).cpu())
                # print(singular_values_p)
                abs_dimension_q = np.sum((singular_values_q) > 0.01)
                abs_dimension_p = np.sum((singular_values_p) > 0.01)
                effective_dimension_q = np.sum((singular_values_q / np.max(singular_values_q)) > 0.01)
                effective_dimension_p = np.sum((singular_values_p / np.max(singular_values_p)) > 0.01)

            # compute the numbers of zero parameters in model
            # zeros_p = 0
            # effective_zeros_p = 0
            # zeros_q = 0
            # effective_zeros_q = 0
            # for param in policy.critic1.preprocess.parameters():
            #     zeros_q += torch.sum(abs(param) < 1e-3).item()
            #     effective_zeros_q += torch.sum((abs(param) / abs(torch.max(param))) < 1e-3).item()
            # for param in policy.actor.preprocess.parameters():
            #     zeros_p += torch.sum(abs(param) < 1e-3).item()
            #     effective_zeros_p += torch.sum((abs(param) / abs(torch.max(param))) < 1e-3).item()
            #
            # total_params_q = sum(p.numel() for p in policy.critic1.preprocess.parameters())
            # total_params_p = sum(p.numel() for p in policy.actor.preprocess.parameters())
            log_data = {"capacity/abs_dimension_q": abs_dimension_q,
                        "capacity/abs_dimension_p": abs_dimension_p,
                        "capacity/effective_dimension_q": effective_dimension_q,
                        "capacity/effective_dimension_p": effective_dimension_p}
            # print("total p:", total_params_p, "total q:", total_params_q)
            logger.write("test/epochs", epochs, log_data)

            # save checkpoint for actor and critic1 state_dict
            checkpoint = {
                "actor": policy.actor.state_dict(),
                "critic1": policy.critic1.state_dict()
            }
            torch.save(checkpoint, os.path.join(path_dict['policy'], "policy_r{}_s{}_{}.pth".format(
                int(args.reset), args.policy_save_seed, epochs)))

        # save the last buffer.
        if epochs == 199:
            buffer.save_hdf5(os.path.join(path_dict['buffer'],
                                          "data_i{}_s{}.hdf5".format(int(args.reset), args.policy_save_seed)))

    if not args.watch:
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
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        )
        pprint.pprint(result)

    # Let's watch its performance!
    # policy.eval()
    # test_envs.seed(args.seed)
    # test_collector.reset()
    # result = test_collector.collect(n_episode=args.test_num, render=args.render)
    # print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    test_sac()
