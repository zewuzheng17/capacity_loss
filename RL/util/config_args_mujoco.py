import argparse
import torch


def get_args():
    update_per_step = 9
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='FingerTurnHard-v1')
    # HopperHop-v1, HumanoidRun-v1, FingerTurnHard-v1, FishSwim-v1, WalkerRun-v1
    parser.add_argument("--algo-name", type=str, default='sac')
    parser.add_argument("--dmc", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[1024, 1024])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--auto-alpha", action="store_false")
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--start-timesteps", type=int, default=5000)  # collect for 10000 times step for starting
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=update_per_step)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )

    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--reset-interval", type=int, default=200000 * update_per_step)  # 200000*step_per_collect
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policy-save-seed", type=int, default=0)
    parser.add_argument("--test_capacity", type=bool, default=True)
    parser.add_argument("--test-capacity-interval", type=int, default=200000 * update_per_step)
    parser.add_argument("--checkpoint-save-interval", type=int, default=5000)

    # test capacity args
    parser.add_argument("--test-capacity-length", type=int, default=20)
    parser.add_argument("--supervised-epoch", type=int, default=10000)
    parser.add_argument("--supervised-data-size", type=int, default=10000)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    return parser.parse_args()
