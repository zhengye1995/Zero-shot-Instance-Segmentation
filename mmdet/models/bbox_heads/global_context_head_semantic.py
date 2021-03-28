import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from ..utils import ConvModule


from mmdet.core import (auto_fp16)

from ..registry import HEADS


@HEADS.register_module
class GlobalContextSemanticHead(nn.Module):
    """Simplest RoI head, with only two fc layers for semantic and
    regression respectively"""

    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 num_convs=3,
                 conv_out_channels=256,
                 num_fcs=1,
                 fc_out_channels=1024,
                 semantic_dims=1024,
                 num_classes=49,
                 conv_cfg=None,
                 norm_cfg=None
                 ):
        super(GlobalContextSemanticHead, self).__init__()

        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        in_channels = self.in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.out_dim = semantic_dims
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, self.last_dim = self._add_conv_fc_branch(num_convs, num_fcs, in_channels)
        # self.fc2 = nn.Linear(self.last_dim, self.out_dim)
        self.final_fc = nn.Linear(self.last_dim, num_classes)



        self.debug_imgs = None

    def _add_conv_fc_branch(self,
                            num_convs,
                            num_fcs,
                            in_channels):
        last_layer_dim = in_channels
        context_convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                context_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
            # add branch specific fc layers
        context_fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= self.roi_feat_area
            for i in range(num_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                context_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return context_convs, context_fcs, last_layer_dim

    def init_weights(self):
        nn.init.normal_(self.final_fc.weight, 0, 0.001)
        nn.init.constant_(self.final_fc.bias, 0)
        # nn.init.normal_(self.fc2.weight, 0, 0.001)
        # nn.init.constant_(self.fc2.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        # x = self.relu(self.fc2(x))
        x = self.final_fc(x) # 1024*49
        return x