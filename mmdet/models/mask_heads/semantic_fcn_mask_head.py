import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class SemanticFCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 semantic_dims=300,
                 seen_class=True,
                 gzsd=False,
                 share_semantic=False,
                 sync_bg=False,
                 voc_path=None,
                 vec_path=None,
                 with_learnable_kernel=True,
                 with_decoder=False,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_ed=dict(type='MSELoss', loss_weight=0.5)):
        super(SemanticFCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))

        self.seen_class = seen_class
        self.gzsd = gzsd
        self.share_semantic = share_semantic
        self.with_learnable_kernel = with_learnable_kernel
        self.with_decoder = with_decoder
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        self.loss_ed = build_loss(loss_ed)
        self.sync_bg=sync_bg

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        self.convT = ConvModule(
                    in_channels,
                    300,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg)
        if self.with_decoder:
            self.dconvT = ConvModule(
                        300,
                        in_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg)
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)


        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None


        if voc_path is not None:
            voc = np.loadtxt(voc_path, dtype='float32', delimiter=',')
        else:
            voc = None

        vec_load = np.loadtxt(vec_path, dtype='float32', delimiter=',')

        vec = vec_load[:, :num_classes]
        vec_unseen = np.concatenate([vec_load[:, 0:1], vec_load[:, num_classes:]], axis=1)
        vec = torch.tensor(vec, dtype=torch.float32)
        if voc is not None:
            voc = torch.tensor(voc, dtype=torch.float32)
        vec_unseen = torch.tensor(vec_unseen, dtype=torch.float32)
        self.vec_unseen = vec_unseen.cuda()
        self.vec = vec.cuda()  # 300*n
        self.conv_vec = nn.Conv2d(300, num_classes, 1, bias=False)

        self.conv_vec.weight.data = torch.unsqueeze(torch.unsqueeze(self.vec.t(), -1), -1)

        if not self.seen_class:
            self.con_vec_t = nn.Conv2d(num_classes, 300, 1, bias=False)
            self.con_vec_t.weight.data = torch.unsqueeze(torch.unsqueeze(self.vec, -1), -1)
            self.conv_vec_unseen = nn.Conv2d(300, vec_unseen.shape[1], 1, bias=False)
            self.conv_vec_unseen.weight.data = torch.unsqueeze(torch.unsqueeze(self.vec_unseen.t(), -1), -1)

        if voc is not None:
            self.voc = voc.cuda()  # 300*66
            self.conv_voc = nn.Conv2d(300, self.voc.size(1), 1, bias=False)
            self.conv_voc.weight.data = torch.unsqueeze(torch.unsqueeze(self.voc.t(), -1), -1)
        else:
            self.voc = None

        self.vec_unseen = vec_unseen.cuda()
        if self.with_learnable_kernel:
            if self.voc is not None:
                self.kernel_semantic = nn.Conv2d(self.voc.size(1), 300, kernel_size=3, padding=1)
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Conv2d(300, self.voc.size(1), kernel_size=3, padding=1)
            else:
                self.kernel_semantic = nn.Conv2d(300, 300, kernel_size=3, padding=1)
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Conv2d(300, 300, kernel_size=3, padding=1)


    def init_weights(self):
        for m in [self.upsample]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        if self.with_learnable_kernel:
            for m in [self.kernel_semantic]:
                if m is None:
                    continue
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        for m in [self.conv_vec]:
            for param in m.parameters():
                param.requires_grad = False
        if self.voc is not None:
            for m in [self.conv_voc]:
                for param in m.parameters():
                    param.requires_grad = False

    @auto_fp16()
    def forward(self, x, bg_vector=None):
        if bg_vector and self.sync_bg:
            with torch.no_grad():
                self.conv_vec.weight.data[0] = bg_vector[0]
                if not self.seen_class:
                    self.conv_vec_unseen.weight.data[0] = bg_vector[0]
        for conv in self.convs:
            conv4_x = conv(x)
        if self.upsample is not None:
            conv4_x = self.upsample(conv4_x)
            if self.upsample_method == 'deconv':
                conv4_x = self.relu(conv4_x)

        # encoder
        x = self.convT(conv4_x)
        if self.voc is not None:
            x = self.conv_voc(x)
        if self.with_learnable_kernel:
            x = self.kernel_semantic(x)

        # decoder
        if self.with_decoder:
            if self.with_learnable_kernel:
                d_x = self.d_kernel_semantic(x)
            d_x = self.dconvT(d_x)
        # classification module
        mask_pred_seen = self.conv_vec(x)
        if not self.seen_class and not self.gzsd:
            mask_pred = self.con_vec_t(mask_pred_seen)
            mask_pred_unseen = self.conv_vec_unseen(mask_pred)
            return mask_pred_unseen
        if self.gzsd:
            mask_pred = self.con_vec_t(mask_pred_seen)
            mask_pred_unseen = self.conv_vec_unseen(mask_pred)
            return mask_pred_seen, mask_pred_unseen
        if not self.with_decoder:
            return mask_pred_seen
        else:
            return mask_pred_seen, conv4_x, d_x


    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels, conv4_x=None, d_x=None):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        if self.with_decoder and conv4_x is not None and d_x is not None:
            loss_encoder_decoder = self.loss_ed(conv4_x, d_x)
            loss['loss_ed'] = loss_encoder_decoder
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not self.gzsd:
            for i in range(bboxes.shape[0]):
                bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
                label = labels[i]
                w = max(bbox[2] - bbox[0] + 1, 1)
                h = max(bbox[3] - bbox[1] + 1, 1)
                if not self.class_agnostic:
                    mask_pred_ = mask_pred[i, label, :, :]
                else:
                    mask_pred_ = mask_pred[i, 0, :, :]
                im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

                bbox_mask = mmcv.imresize(mask_pred_, (w, h))
                bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                    np.uint8)
                im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
                rle = mask_util.encode(
                    np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                cls_segms[label - 1].append(rle)

        if self.gzsd:
            if self.num_classes == 49:
                cls_segms = [[] for _ in range(48+17)]

            elif self.num_classes == 66:
                cls_segms = [[] for _ in range(65 + 15)]

            for i in range(bboxes.shape[0]):
                bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
                label = labels[i]
                w = max(bbox[2] - bbox[0] + 1, 1)
                h = max(bbox[3] - bbox[1] + 1, 1)
                temp_label = label
                if mask_pred.shape[1] == 66:
                    if label >= 66:
                        continue
                if mask_pred.shape[1] == 16:
                    if label < 66:
                        continue
                    label -= 65
                if mask_pred.shape[1] == 48:
                    if label >= 48:
                        continue
                if mask_pred.shape[1] == 18:
                    if label < 48:
                        continue
                    label -= 47
                if not self.class_agnostic:
                    mask_pred_ = mask_pred[i, label, :, :]
                else:
                    mask_pred_ = mask_pred[i, 0, :, :]
                im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

                bbox_mask = mmcv.imresize(mask_pred_, (w, h))
                bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                    np.uint8)
                im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
                rle = mask_util.encode(
                    np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                cls_segms[temp_label - 1].append(rle)

        return cls_segms



