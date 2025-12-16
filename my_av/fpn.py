import torch.nn as nn
import torch.nn.functional as F

from collections import deque


class FPN(nn.Module):
    def __init__(self,
                 in_channels_list: list,
                 out_channels: int,
                 upsample_cfg: dict = dict(mode="nearest")):
        """
        Args:
            in_channels_list (list[int]): list of input channel sizes for each feature map
            out_channels (int): number of output channels for every P-layer
            upsample_cfg (dict): configuration for upsampling layers
        """
        super(FPN, self).__init__()

        self.upsample_cfg = upsample_cfg

        # 1×1 lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        # 3×3 smoothing convolutions
        self.smooth_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.smooth_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): feature maps from the backbone (B, C_i, H_i, W_i)
        Returns:
            list[Tensor]: pyramid feature maps (B, out_channels, H_i, W_i)
        """
        # Build laterals
        laterals = []
        for i, x in enumerate(inputs):
            laterals.append(self.lateral_convs[i](x))

        # Top-down pathway
        top_downs = deque([laterals[-1]])
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(top_downs[0], size=laterals[i].shape[-2:], **self.upsample_cfg)
            top_downs.appendleft(laterals[i] + upsampled)
 
        # Smoothing
        Ps = []
        for i in range(len(top_downs)):
            Ps.append(self.smooth_convs[i](top_downs[i]))

        return Ps
