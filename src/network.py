from typing import List
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils


def _get_convolution_net(
    in_channels: int,
    history_len: int = 1,
    channels: List = None,
    use_bias: bool = False,
    use_groups: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    channels = channels or [16, 32, 16]
    activation_fn = torch.nn.__dict__[activation]

    def _get_block(**conv_params):
        layers = [nn.Conv2d(**conv_params)]
        if use_normalization:
            layers.append(nn.InstanceNorm2d(conv_params["out_channels"]))
        if use_dropout:
            layers.append(nn.Dropout2d(p=0.1))
        layers.append(activation_fn(inplace=True))
        layers.append(nn.MaxPool2d(2))
        return layers

    channels.insert(0, history_len * in_channels)
    params = []
    for i, (in_channels, out_channels) in enumerate(utils.pairwise(channels)):
        num_groups = 1
        if use_groups:
            num_groups = history_len if i == 0 else 4
        params.append(
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "bias": use_bias,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "groups": num_groups,
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.initialization.get_optimal_inner_init(activation_fn))
    return net


def _get_linear_net(
    in_features: int,
    history_len: int = 1,
    features: List = None,
    use_bias: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    features = features or [64, 128, 64]
    activation_fn = torch.nn.__dict__[activation]

    def _get_block(**linear_params):
        layers = [nn.Linear(**linear_params)]
        if use_normalization:
            layers.append(nn.LayerNorm(linear_params["out_features"]))
        if use_dropout:
            layers.append(nn.Dropout(p=0.1))
        layers.append(activation_fn(inplace=True))
        return layers

    features.insert(0, history_len * in_features)
    params = []
    for i, (in_features, out_features) in enumerate(utils.pairwise(features)):
        params.append(
            {
                "in_features": in_features,
                "out_features": out_features,
                "bias": use_bias,
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.initialization.get_optimal_inner_init(activation_fn))

    return net


# calculating CNN  output elements
def CNN_output(in_channels, im_width, im_height, image_net):
    input_shape: tuple = (in_channels, im_width, im_height)
    CNN_input = torch.Tensor(torch.randn((1,) + input_shape))
    CNN_output = image_net(CNN_input)
    CNN_out_features = CNN_output.nelement()
    return CNN_out_features


class StateNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        image_net: nn.Module = None,
    ):
        super().__init__()
        self.main_net = main_net
        self.image_net = image_net

    def forward(self, state):
        image = state["cam_image"]

        batch_size, _, h, w, c = image.shape
        # Here we need to change from NHWC to NCHW
        image = image.contiguous().permute(0, 1, 4, 2, 3).view(batch_size, -1, h, w)
        image_features = self.image_net(image)
        image_features = image_features.contiguous().view(batch_size, -1)

        x = image_features
        x = self.main_net(x)

        return x

    @classmethod
    def get_from_params(
        cls,
        im_width=None,
        im_height=None,
        in_channels=None,
        image_net_params=None,
        main_net_params=None,
    ) -> "StateNet":
        image_net_params = deepcopy(image_net_params)
        main_net_params = deepcopy(main_net_params)

        image_net = _get_convolution_net(in_channels, **image_net_params)

        # calculating CNN output elements
        CNN_out_features = CNN_output(in_channels, im_width, im_height, image_net)

        main_net_in_features = CNN_out_features

        main_net = _get_linear_net(main_net_in_features,**main_net_params)

        net = cls(
            image_net=image_net,
            main_net=main_net
        )

        return net


class StateActionNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        image_net: nn.Module = None,
        action_net: nn.Module = None
    ):
        super().__init__()
        self.main_net = main_net
        self.image_net = image_net
        self.action_net = action_net

    def forward(self, state, action):
        # link features state with enabled
        image = state["cam_image"]

        batch_size, _, h, w, c = image.shape
        # Here we need to change from NHWC to NCHW
        image = image.contiguous().permute(0, 1, 4, 2, 3).view(batch_size, -1, h, w)
        image_features = self.image_net(image)
        image_features = image_features.contiguous().view(batch_size, -1)

        action = self.action_net(action)

        x = torch.cat([image_features, action], dim=1)

        x = self.main_net(x)

        return x

    @classmethod
    def get_from_params(
        cls,
        im_width = None,
        im_height = None,
        in_channels = None,
        action_in_features = None,
        image_net_params=None,
        action_net_params=None,
        main_net_params=None,
    ) -> "StateActionNet":

        image_net_params = deepcopy(image_net_params)
        action_net_params = deepcopy(action_net_params)
        main_net_params = deepcopy(main_net_params)

        image_net = _get_convolution_net(in_channels, **image_net_params)

        # calculating CNN output elements
        CNN_out_features = CNN_output(in_channels, im_width, im_height, image_net)

        action_net = _get_linear_net(action_in_features, **action_net_params)
        action_net_out_features = action_net_params["features"][-1]

        main_net_in_features = CNN_out_features + action_net_out_features

        main_net_params["in_features"] = main_net_in_features
        main_net = _get_linear_net(**main_net_params)

        net = cls(
            image_net=image_net,
            action_net=action_net,
            main_net=main_net
        )

        return net
