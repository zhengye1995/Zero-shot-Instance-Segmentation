import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from ..registry import HEADS
from .anchor_semantic_head import AnchorSemanticHead


@HEADS.register_module
class RPNSemanticHead(AnchorSemanticHead):

    def __init__(self, in_channels, freeze=False, **kwargs):
        self.freeze=freeze
        super(RPNSemanticHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.semantic_dims, 1)
        # self.rpn_cls = nn.Conv2d(self.feat_channels,
        #                          self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        if self.freeze:
            for m in [self.rpn_conv, self.rpn_cls, self.rpn_reg]:
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        # checkpoint_temp = torch.load('./work_dirs/cascade_rcnn_r50_fpn_1x_48_seen_classes_semantic/latest.pth',
        #                              map_location='cpu')
        # checkpoint_temp = torch.load('./work_dirs/cascade_rcnn_r50_fpn_1x_65seen_classes_semantic/latest.pth',
        #                              map_location='cpu')
        # TODO
        # kernel_semantic_weight = checkpoint_temp['state_dict']['bbox_head.2.kernel_semantic.weight']
        # kernel_semantic_bias = checkpoint_temp['state_dict']['bbox_head.2.kernel_semantic.bias']
        #
        # self.kernel_semantic.weight = torch.nn.Parameter(kernel_semantic_weight)
        # self.kernel_semantic.bias = torch.nn.Parameter(kernel_semantic_bias)
        # for param in self.kernel_semantic.parameters():
        #     param.requires_grad = False
        with torch.no_grad():
            self.vec_bg.weight.data[0] = self.vec_bg_weight


    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        # B C(900) W H
        rpn_cls_score = self.rpn_cls(x)
        # B 3 W H 300
        B, C, H, W = rpn_cls_score.size()
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
        rpn_cls_score = rpn_cls_score.view(B, H, W, 3, self.semantic_dims)
        rpn_cls_score = rpn_cls_score.contiguous().view(B*H*W*3, self.semantic_dims)
        # rpn_cls_score = rpn_cls_score.view(rpn_cls_score.size(0), self.num_anchors, self.semantic_dims,
        #                                    rpn_cls_score.size(2), rpn_cls_score.size(3))
        # rpn_cls_score = rpn_cls_score.permute(0, 1, 3, 4, 2)

        # rpn_cls_score = rpn_cls_score.contiguous().view(rpn_cls_score.size(0)*rpn_cls_score.size(1)*rpn_cls_score.size(2)*rpn_cls_score.size(3),
        #                                    rpn_cls_score.size(4)) # xxx*300
        if self.voc:
            semantic_score = self.kernel_semantic(self.voc) #300*300
            semantic_score = self.vec_bg(semantic_score) # 300*2
            semantic_score = torch.tanh(semantic_score) # 300*2
            # rpn_cls_score = self.vec_bg(rpn_cls_score)
            rpn_cls_score = torch.mm(rpn_cls_score, semantic_score) # b*h*w*3 * 2
            # rpn_cls_score = rpn_cls_score.view(B, self.num_anchors*2, H, W)
        else:
            tempinput = torch.ones(self.semantic_dims, self.semantic_dims).cuda()
            semantic_score = self.kernel_semantic(tempinput)  # 300*300
            semantic_score = self.vec_bg(semantic_score)  # 300*2
            semantic_score = torch.tanh(semantic_score)  # 300*2
            # rpn_cls_score = self.vec_bg(rpn_cls_score)
            rpn_cls_score = torch.mm(rpn_cls_score, semantic_score)  # b*h*w*3 * 2
        rpn_cls_score = rpn_cls_score.view(B, H, W, 3*2)
        rpn_cls_score = rpn_cls_score.permute(0, 3, 1, 2)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(RPNSemanticHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
