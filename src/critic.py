from typing import Dict

import torch

from catalyst_rl.rl.agent.head import ValueHead
from catalyst_rl.rl.core import CriticSpec, EnvironmentSpec

from .network import StateActionNet


class UR5StateActionCritic(CriticSpec):
    def __init__(self, state_action_net: StateActionNet, head_net: ValueHead):
        super().__init__()
        self.state_action_net = state_action_net
        self.head_net = head_net

    @property
    def num_outputs(self) -> int:
        return self.head_net.out_features

    @property
    def num_atoms(self) -> int:
        return self.head_net.num_atoms

    @property
    def distribution(self) -> str:
        return self.head_net.distribution

    @property
    def values_range(self) -> tuple:
        return self.head_net.values_range

    @property
    def num_heads(self) -> int:
        return self.head_net.num_heads

    @property
    def hyperbolic_constant(self) -> float:
        return self.head_net.hyperbolic_constant

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = self.state_action_net(state, action)
        x = self.head_net(x)
        return x

    @classmethod
    def get_from_params(
        cls,
        state_action_net_params: Dict,
        value_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        im_width, im_height = env_spec.observation_space["cam_image"].shape[-2], \
                              env_spec.observation_space["cam_image"].shape[-3]
        in_channels = env_spec.observation_space["cam_image"].shape[-1]
        # action net input
        action_in_features = env_spec.action_space.shape[0]
        # state action critic network
        state_action_net = StateActionNet.get_from_params(
            im_width, im_height, in_channels, action_in_features,  **state_action_net_params
        )
        head_net = ValueHead(**value_head_params)

        net = cls(state_action_net=state_action_net, head_net=head_net)
        print('---------Critic--------\n{}'.format(net))
        return net
