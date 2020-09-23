from typing import Dict
from gym.spaces import Box, Discrete

import torch

from catalyst_rl.rl.agent.head import PolicyHead
from catalyst_rl.rl.core import ActorSpec, EnvironmentSpec

from .network import StateNet


class UR5Actor(ActorSpec):
    def __init__(self, state_net: StateNet, head_net: PolicyHead):
        super().__init__()
        self.state_net = state_net
        self.head_net = head_net

    @property
    def policy_type(self) -> str:
        return self.head_net.policy_type

    def forward(self, state: torch.Tensor, logprob=False, deterministic=False):
        x = self.state_net(state)
        x = self.head_net(x, logprob, deterministic)
        return x

    @classmethod
    def get_from_params(
            cls,
            state_net_params: Dict,
            policy_head_params: Dict,
            env_spec: EnvironmentSpec,
    ):
        action_space = env_spec.action_space
        if isinstance(action_space, Box):
            policy_head_params["out_features"] = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            policy_head_params["out_features"] = action_space.n
        else:
            raise NotImplementedError()

        im_width, im_height = env_spec.observation_space["cam_image"].shape[-2], \
                              env_spec.observation_space["cam_image"].shape[-3]
        in_channels = env_spec.observation_space["cam_image"].shape[-1]

        # Actor Network
        state_net = StateNet.get_from_params(im_width, im_height, in_channels, **state_net_params)
        head_net = PolicyHead(**policy_head_params)

        net = cls(state_net=state_net, head_net=head_net)
        print('-----------Actor------------\n{}'.format(net))
        return net
