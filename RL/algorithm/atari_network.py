from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from tianshou.utils.net.discrete import NoisyLinear

'''
This files contain the cultivation of specific deep network
class DQN consist of a base model that is a basis structure for atari environments.
Then, the other network, like rainbow and qrdqn, are built on top of DQN class.
'''


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            device: Union[str, int, torch.device] = "cpu",
            features_only: bool = False,
            output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, 512), nn.ReLU(inplace=True),
                nn.Linear(512, np.prod(action_shape))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, output_dim),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


class C51(DQN):
    """Reference: A distributional perspective on reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            num_atoms: int = 51,
            device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_atoms], device)
        self.num_atoms = num_atoms

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.num_atoms).softmax(dim=-1)
        obs = obs.view(-1, self.action_num, self.num_atoms)
        return obs, state


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            num_atoms: int = 51,
            noisy_std: float = 0.5,
            device: Union[str, int, torch.device] = "cpu",
            is_dueling: bool = True,
            is_noisy: bool = True,
            add_infer: bool = False,
            infer_multi_head_num: int = 10,
            infer_output_dim: int = 10,
    ) -> None:
        super().__init__(c, h, w, action_shape, device, features_only=True)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms
        self.add_infer = add_infer
        self.infer_multi_head_num = infer_multi_head_num
        self.infer_output_dim = infer_output_dim

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            else:
                return nn.Linear(x, y)

        self.q_representation = linear(self.output_dim, 512)
        self.Q = nn.Sequential(
            nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms)
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512), nn.ReLU(inplace=True),
                linear(512, self.num_atoms)
            )
        self.output_dim = self.action_num * self.num_atoms

        if self.add_infer:
            self.multi_head_output = nn.Linear(512, self.infer_multi_head_num * self.infer_output_dim)

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any, any, any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        q_representation = self.q_representation(obs)
        q = self.Q(q_representation)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q

        if self.add_infer:
            multi_head_output = self.multi_head_output(q_representation)
            multi_head_output = multi_head_output.view(-1,  self.infer_output_dim, self.infer_multi_head_num)
        else:
            multi_head_output = None
        # perform softmax operation over the output of value of action
        probs = logits.softmax(dim=2)
        return probs, state, q_representation, multi_head_output


class QRDQN(DQN):
    """Reference: Distributional Reinforcement Learning with Quantile \
    Regression.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            num_quantiles: int = 200,
            device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_quantiles], device)
        self.num_quantiles = num_quantiles

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.action_num, self.num_quantiles)
        return obs, state

