from __future__ import division

import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import RPNTestMixin
from ..losses import LSoftmaxLinear, CenterLoss


@DETECTORS.register_module
class CascadeRCNN(BaseDetector, RPNTestMixin):

    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 semantic_info_flow=False,
                 use_lfostmax=False,
                 gzsd_mode=False,
                 use_center_loss=False,
                 context_roi=None,
                 context_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(CascadeRCNN, self).__init__()

        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))

        if mask_head is not None:
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_head) == self.num_stages
            for head in mask_head:
                self.mask_head.append(builder.build_head(head))
            if mask_roi_extractor is not None:
                self.share_roi_extractor = False
                self.mask_roi_extractor = nn.ModuleList()
                if not isinstance(mask_roi_extractor, list):
                    mask_roi_extractor = [
                        mask_roi_extractor for _ in range(num_stages)
                    ]
                assert len(mask_roi_extractor) == self.num_stages
                for roi_extractor in mask_roi_extractor:
                    self.mask_roi_extractor.append(
                        builder.build_roi_extractor(roi_extractor))
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if context_roi is not None:
            self.context_roi = builder.build_roi_extractor(context_roi)
        else:
            self.context_roi = None
        if context_head is not None:
            self.context_head = builder.build_head(context_head)
        else:
            self.context_head = None
        self.semantic_info_flow = semantic_info_flow
        self.use_lsoftmax = use_lfostmax
        if self.use_lsoftmax:
            self.lsoftmax = LSoftmaxLinear(49, 49, margin=3)
        self.use_center_loss = use_center_loss
        if self.use_center_loss:
            self.center_loss = CenterLoss(num_classes=49, feat_dim=300)
        self.gzsd_mode = gzsd_mode
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CascadeRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox heads
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_feats = self.bbox_roi_extractor[i](
                    x[:self.bbox_roi_extractor[i].num_inputs], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)
                cls_score, bbox_pred = self.bbox_head[i](bbox_feats)
                outs = outs + (cls_score, bbox_pred)
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_feats = self.mask_roi_extractor[i](
                    x[:self.mask_roi_extractor[i].num_inputs], mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head[i](mask_feats)
                outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(
                    rcnn_train_cfg.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])

            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            # if self.context_roi is not None:
                # bbox_feats = bbox_feats + context_feats

            if self.context_roi is not None:
                img_rois = bbox2roi([torch.tensor([[0, img.size(3), 0, img.size(2)]],
                                                  dtype=torch.float).repeat(rois.size(0), 1).cuda()])
                context_feats = self.context_roi(x[:self.context_roi.num_inputs],
                                                 img_rois)
                context_feats = self.context_head(context_feats)
            else:
                context_feats = None

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            if self.semantic_info_flow:
                # TODO
                last_feat = None
                for j in range(i):
                    cls_score, bbox_pred, last_feat = self.bbox_head[j](
                        bbox_feats, last_feat, context_feats, return_feats=True)
                cls_score, bbox_pred = bbox_head(bbox_feats, last_feat, context_feats, return_feats=False)
            else:
                # TODO ZSD
                if not self.use_center_loss:
                    cls_score, bbox_pred = bbox_head(bbox_feats, None, context_feats, return_feats=False,
                                                     resturn_center_feats=False)
                else:
                    cls_score, bbox_pred, center_feat = bbox_head(bbox_feats, None, context_feats, return_feats=False,
                                                                  resturn_center_feats=True)
                # todo OD
                # cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                                gt_labels, rcnn_train_cfg)
            if self.use_lsoftmax:
                cls_score = self.lsoftmax(cls_score, bbox_targets[0])
            if self.use_center_loss:
                center_loss = self.center_loss(center_feat, bbox_targets[0])
                losses['s{}.{}'.format(i, 'center_loss')] = (center_loss*lw) * 0.1
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if not self.share_roi_extractor:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    mask_feats = mask_roi_extractor(
                        x[:mask_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                else:
                    # reuse positive bbox feats
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(
                                res.pos_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(
                                res.neg_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds]
                mask_head = self.mask_head[i]
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                                    rcnn_train_cfg)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Run inference on a single image.

        Args:
            img (Tensor): must be in shape (N, C, H, W)
            img_meta (list[dict]): a list with one dictionary element.
                See `mmdet/datasets/pipelines/formatting.py:Collect` for
                details of meta dicts.
            proposals : if specified overrides rpn proposals
            rescale (bool): if True returns boxes in original image space

        Returns:
            dict: results
        """
        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)


        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            if self.context_roi is not None:
                img_rois = bbox2roi([torch.tensor([[0, img.size(3), 0, img.size(2)]],
                                                  dtype=torch.float).repeat(rois.size(0), 1).cuda()])
                context_feats = self.context_roi(x[:self.context_roi.num_inputs],
                                                 img_rois)
                context_feats = self.context_head(context_feats)
            else:
                context_feats = None

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            if self.semantic_info_flow:
                # TODO
                last_feat = None
                for j in range(i):
                    # cls_score, bbox_pred, last_feat = self.bbox_head[i](bbox_feats, last_feat, context_feats,
                    #                                                    return_feats=True)
                    cls_score, bbox_pred, last_feat = self.bbox_head[j](bbox_feats, last_feat, context_feats,
                                                                       return_feats=True)
                cls_score, bbox_pred = bbox_head(bbox_feats, last_feat, context_feats, return_feats=False)
            else:
                # TODO ZSD
                cls_score, bbox_pred = bbox_head(bbox_feats, None, context_feats, return_feats=False)
                # OD
                # cls_score, bbox_pred = bbox_head(bbox_feats)

            # if self.context_roi is not None:
            #     cls_score, bbox_pred = bbox_head(bbox_feats, context_feats)
            # else:
            #     cls_score, bbox_pred = bbox_head(bbox_feats)
            if self.use_lsoftmax:
                cls_score = self.lsoftmax(cls_score)

            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        mask_classes = mask_head.num_classes - 1
                        segm_result = [[] for _ in range(mask_classes)]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] *
                            scale_factor if rescale else det_bboxes)
                        mask_rois = bbox2roi([_bboxes])
                        mask_feats = mask_roi_extractor(
                            x[:len(mask_roi_extractor.featmap_strides)],
                            mask_rois)
                        if self.with_shared_head:
                            mask_feats = self.shared_head(mask_feats, i)
                        mask_pred = mask_head(mask_feats)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)

        # TODO GZSD
        if self.gzsd_mode:
            if self.bbox_head[-1].num_classes == 66:
                results = bbox2result(det_bboxes, det_labels, 65+15+1)
            else:
                results = bbox2result(det_bboxes, det_labels, 48+17+1)

            return results

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        # TODO output num_classes  open these line when unseen inference
        # bbox_result = bbox2result(det_bboxes, det_labels, 18)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                if isinstance(scale_factor, float):  # aspect ratio fixed
                    _bboxes = (
                        det_bboxes[:, :4] *
                        scale_factor if rescale else det_bboxes)
                else:
                    _bboxes = (
                        det_bboxes[:, :4] *
                        torch.from_numpy(scale_factor).to(det_bboxes.device)
                        if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_feats = mask_roi_extractor(
                        x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                    mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'])
            else:
                results = ms_bbox_result['ensemble']
        else:
            if self.with_mask:
                results = {
                    stage: (ms_bbox_result[stage], ms_segm_result[stage])
                    for stage in ms_bbox_result
                }
            else:
                results = ms_bbox_result

        return results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn
        aug_bboxes = []
        aug_scores = []
        if self.gzsd_mode:
            seen_aug_bboxes = []
            seen_aug_scores = []
            unseen_aug_bboxes = []
            unseen_aug_scores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_roi_extractor = self.bbox_roi_extractor[i]
                bbox_head = self.bbox_head[i]

                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)

                if self.semantic_info_flow:
                    # TODO
                    last_feat = None
                    for j in range(i):
                        # cls_score, bbox_pred, last_feat = self.bbox_head[i](bbox_feats, last_feat, context_feats,
                        #                                                    return_feats=True)
                        cls_score, bbox_pred, last_feat = self.bbox_head[j](bbox_feats, last_feat, None,
                                                                            return_feats=True)
                    cls_score, bbox_pred = bbox_head(bbox_feats, last_feat, None, return_feats=False)
                else:
                    # TODO ZSD
                    cls_score, bbox_pred = bbox_head(bbox_feats, None, None, return_feats=False)
                    # OD
                    # cls_score, bbox_pred = bbox_head(bbox_feats)

                # cls_score, bbox_pred = bbox_head(bbox_feats)
                ms_scores.append(cls_score)

                if i < self.num_stages - 1:
                    bbox_label = cls_score.argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label,
                                                      bbox_pred, img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)

            if self.gzsd_mode:
                seen_aug_bboxes.append(bboxes[0])
                seen_aug_scores.append(scores[0])
                unseen_aug_bboxes.append(bboxes[1])
                unseen_aug_scores.append(scores[1])
                seen_merged_bboxes, seen_merged_scores = merge_aug_bboxes(
                    seen_aug_bboxes, seen_aug_scores, img_metas, rcnn_test_cfg)
                unseen_merged_bboxes, unseen_merged_scores = merge_aug_bboxes(
                    unseen_aug_bboxes, unseen_aug_scores, img_metas, rcnn_test_cfg)
                seen_det_bboxes, seen_det_labels = multiclass_nms(seen_merged_bboxes, seen_merged_scores,
                                                        0.2,
                                                        rcnn_test_cfg.nms,
                                                        rcnn_test_cfg.max_per_img)
                unseen_det_bboxes, unseen_det_labels = multiclass_nms(unseen_merged_bboxes, unseen_merged_scores,
                                                                  0.05,
                                                                  rcnn_test_cfg.nms,
                                                                  rcnn_test_cfg.max_per_img)
                unseen_det_labels += (self.bbox_head[-1].num_classes - 1)

                det_bboxes = torch.cat([seen_det_bboxes, unseen_det_bboxes], dim=0)
                det_labels = torch.cat([seen_det_labels, unseen_det_labels], dim=0)

                if self.bbox_head[-1].num_classes == 66:
                    results = bbox2result(det_bboxes, det_labels, 65 + 15 + 1)
                else:
                    results = bbox2result(det_bboxes, det_labels, 48 + 17 + 1)

                return results

            aug_bboxes.append(bboxes)
            aug_scores.append(scores)


        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes -
                                              1)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(self.extract_feats(imgs), img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_feats = self.mask_roi_extractor[i](
                            x[:len(self.mask_roi_extractor[i].featmap_strides
                                   )], mask_rois)
                        if self.with_shared_head:
                            mask_feats = self.shared_head(mask_feats)
                        mask_pred = self.mask_head[i](mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg.rcnn)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return bbox_result, segm_result
        else:
            return bbox_result

    def show_result(self, data, result, **kwargs):
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        super(CascadeRCNN, self).show_result(data, result, **kwargs)
