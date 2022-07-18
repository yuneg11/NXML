import torch
import torch.nn as nn

from nxcl.config import ConfigDict as CfgNode
from ...layers import *


class VGG16(nn.Module):

    def __init__(
            self,
            channels: int,
            in_planes: int,
            conv: nn.Module = Conv2d,
            norm: nn.Module = BatchNorm2d,
            relu: nn.Module = ReLU,
            **kwargs,
        ) -> None:
        super().__init__()
        self.channels  = channels
        self.in_planes = in_planes
        self.conv      = conv
        self.norm      = norm
        self.relu      = relu

        self.layers = nn.Sequential(
            nn.Sequential(
                self.conv(in_channels=self.channels, out_channels=self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=self.in_planes),
                self.relu(),
                self.conv(in_channels=self.in_planes, out_channels=self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=self.in_planes),
                self.relu(),
                MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                self.conv(in_channels=self.in_planes, out_channels=2*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=2*self.in_planes),
                self.relu(),
                self.conv(in_channels=2*self.in_planes, out_channels=2*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=2*self.in_planes),
                self.relu(),
                MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                self.conv(in_channels=2*self.in_planes, out_channels=4*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=4*self.in_planes),
                self.relu(),
                self.conv(in_channels=4*self.in_planes, out_channels=4*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=4*self.in_planes),
                self.relu(),
                self.conv(in_channels=4*self.in_planes, out_channels=4*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=4*self.in_planes),
                self.relu(),
                MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                self.conv(in_channels=4*self.in_planes, out_channels=8*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=8*self.in_planes),
                self.relu(),
                self.conv(in_channels=8*self.in_planes, out_channels=8*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=8*self.in_planes),
                self.relu(),
                self.conv(in_channels=8*self.in_planes, out_channels=8*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=8*self.in_planes),
                self.relu(),
                MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                self.conv(in_channels=8*self.in_planes, out_channels=8*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=8*self.in_planes),
                self.relu(),
                self.conv(in_channels=8*self.in_planes, out_channels=8*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=8*self.in_planes),
                self.relu(),
                self.conv(in_channels=8*self.in_planes, out_channels=8*self.in_planes,
                          kernel_size=3, stride=1, padding=1, **kwargs),
                self.norm(num_features=8*self.in_planes),
                self.relu(),
                MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                Flatten(),

            )
        )
