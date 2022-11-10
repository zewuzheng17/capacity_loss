import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    # env parameters
    parser.add_argument("--task", type=str, default="UpNDown-v5") # UpNDown-v5
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)

    # rl parameters
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.)
    parser.add_argument("--v-max", type=float, default=10.)
    parser.add_argument("--noisy-std", type=float, default=0.1)
    parser.add_argument("--no-dueling", action="store_true", default=False) # always no dueling
    parser.add_argument("--no-noisy", action="store_true", default=False)
    parser.add_argument("--no-priority", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.)
    parser.add_argument("--beta-anneal-step", type=int, default=5000000)
    parser.add_argument("--no-weight-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=500000)
    parser.add_argument("--step-per-collect", type=int, default=20)
    parser.add_argument("--update-per-step", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=5)

    # logging and rendering, resume
    parser.add_argument("--logdir", type=str, default="log/")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-checkpoint-name", type=str, default=None)
    parser.add_argument("--resume-buffer-name", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)

    # supervised task parameters, and add infer parameters
    parser.add_argument("--test-capacity-length", type=int, default=50)
    parser.add_argument("--supervised-epoch", type=int, default=10000)
    parser.add_argument("--supervised-data-size", type=int, default=1024)
    parser.add_argument("--checkpoint-save-interval", type=int, default=2000000)
    parser.add_argument("--add-infer", action='store_true')
    parser.add_argument("--infer-multi-head-num", type=int, default=10)
    parser.add_argument("--infer-output-dim", type=int, default=1)
    parser.add_argument("--infer_gradient_scale", type=float, default=0.1) # seed 0 (0.1,500) seed 1 (1, 100)
    parser.add_argument("--infer-target_scale", type=float, default=100.)
    parser.add_argument("--grad-norm", action="store_true")
    parser.add_argument("--collect-test-statistics", action="store_false")
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--global-grad-norm", action = "store_true")
    parser.add_argument("--reset-policy", action="store_true")
    parser.add_argument("--reset-policy-interval", type=int, default=400000) # reset interval for gradient steps
    parser.add_argument("--policy-save-seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
