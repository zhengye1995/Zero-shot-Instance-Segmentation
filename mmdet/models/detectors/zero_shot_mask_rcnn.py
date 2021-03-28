from ..registry import DETECTORS
from .zero_shot_two_stage import ZeroShotTwoStageDetector


@DETECTORS.register_module
class ZeroShotMaskRCNN(ZeroShotTwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 mask_with_decoder=False,
                 bbox_with_decoder=False,
                 bbox_sync_bg=False,
                 mask_sync_bg=False,
                 gzsd_mode=False):
        super(ZeroShotMaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            bbox_with_decoder=bbox_with_decoder,
            mask_with_decoder=mask_with_decoder,
            bbox_sync_bg=bbox_sync_bg,
            mask_sync_bg=mask_sync_bg,
            gzsd_mode=gzsd_mode)
