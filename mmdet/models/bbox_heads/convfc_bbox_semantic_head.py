import torch.nn as nn
import torch

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head_semantic import BBoxSemanticHead


@HEADS.register_module
class ConvFCSemanticBBoxHead(BBoxSemanticHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_semantic_convs=0,
                 num_semantic_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 semantic_dims=300,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCSemanticBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_semantic_convs +
                num_semantic_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_semantic_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_semantic:
            assert num_semantic_convs == 0 and num_semantic_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_semantic_convs = num_semantic_convs
        self.num_semantic_fcs = num_semantic_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add semantic specific branch
        self.semantic_convs, self.semantic_fcs, self.semantic_last_dim = \
            self._add_conv_fc_branch(
                self.num_semantic_convs, self.num_semantic_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_semantic_fcs == 0:
                self.semantic_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_semantic and fc_reg since input channels are changed
        if self.with_semantic:
            self.fc_semantic = nn.Linear(self.semantic_last_dim, semantic_dims)
            if self.with_decoder:
                self.d_fc_semantic = nn.Linear(semantic_dims, self.semantic_last_dim)
            if self.voc is not None:
                self.kernel_semantic = nn.Linear(self.voc.shape[1], self.vec.shape[0])  # n*300
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Linear(self.vec.shape[0], self.voc.shape[1])  # n*300
            else:
                self.kernel_semantic = nn.Linear(self.vec.shape[1], self.vec.shape[1])
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Linear(self.vec.shape[1], self.vec.shape[1])  # n*300

        if self.with_reg and self.reg_with_semantic:
            self.fc_reg_sem = nn.Linear(self.reg_last_dim, semantic_dims)
            if not self.share_semantic:
                self.kernel_semantic_reg = nn.Linear(self.voc.shape[1], self.vec.shape[0])
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.num_classes, out_dim_reg)

        if self.with_reg and not self.reg_with_semantic:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        self.fc_res = nn.Linear(self.vec.shape[0], self.vec.shape[0])
        # self.fc_res = nn.Linear(self.semantic_last_dim, self.vec.shape[0])

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCSemanticBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.semantic_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, res_feats=None, context_feats=None, return_feats=False, resturn_center_feats=False, bg_vector=None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_semantic = x
        x_reg = x

        for conv in self.semantic_convs:
            x_semantic = conv(x_semantic)
        if x_semantic.dim() > 2:
            if self.with_avg_pool:
                x_semantic = self.avg_pool(x_semantic)
            x_semantic = x_semantic.view(x_semantic.size(0), -1)
        for fc in self.semantic_fcs:
            x_semantic = self.relu(fc(x_semantic))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if self.with_semantic:
            semantic_feature = self.fc_semantic(x_semantic)
            if self.sync_bg:
                with torch.no_grad():
                    self.vec[:, 0] = bg_vector
                    if not self.seen_class:
                        self.vec_unseen[:, 0] = bg_vector
            if self.voc is not None:

                semantic_score = torch.mm(semantic_feature, self.voc)
                if self.semantic_norm:
                    semantic_score_norm = torch.norm(semantic_score, p=2, dim=1).unsqueeze(1).expand_as(semantic_score)
                    semantic_score = semantic_score.div(semantic_score_norm + 1e-5)
                    temp_norm = torch.norm(self.kernel_semantic.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.kernel_semantic.weight.data)
                    self.kernel_semantic.weight.data = self.kernel_semantic.weight.data.div(temp_norm + 1e-5)
                    semantic_score = self.kernel_semantic(semantic_score) * 20.0
                else:
                    semantic_score = self.kernel_semantic(semantic_score)
                if self.with_decoder:
                    d_semantic_score = self.d_kernel_semantic(semantic_score)
                    d_semantic_feature = torch.mm(d_semantic_score, self.voc.t())
                    d_semantic_feature = self.d_fc_semantic(d_semantic_feature)

                semantic_score = torch.mm(semantic_score, self.vec)
            else:
                semantic_score = self.kernel_semantic(self.vec)
                semantic_score = torch.tanh(semantic_score)
                semantic_score = torch.mm(semantic_feature, semantic_score)
        else:
            semantic_score = None
        if self.with_reg and not self.reg_with_semantic:
            bbox_pred = self.fc_reg(x_reg)
        elif self.with_reg and self.reg_with_semantic:
            semantic_reg_feature = self.fc_reg_sem(x_reg)
            if not self.share_semantic:
                semantic_reg_score = torch.mm(self.kernel_semantic_reg(self.voc), self.vec)
            else:
                semantic_reg_score = torch.mm(self.kernel_semantic(self.voc), self.vec)
            semantic_reg_score = torch.tanh(semantic_reg_score)
            semantic_reg_score = torch.mm(semantic_reg_feature, semantic_reg_score)
            bbox_pred = self.fc_reg(semantic_reg_score)
        else:
            bbox_pred = None
        if self.with_decoder:
            return semantic_score, bbox_pred, x_semantic, d_semantic_feature
        else:
            return semantic_score, bbox_pred


@HEADS.register_module
class SharedFCSemanticBBoxHead(ConvFCSemanticBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCSemanticBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_semantic_convs=0,
            num_semantic_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
