from detectron2.modeling.proposal_generator import RPN, PROPOSAL_GENERATOR_REGISTRY, build_rpn_head
from detectron2.modeling.proposal_generator.rpn import find_top_rpn_proposals
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.matcher import Matcher
from detectron2.config import configurable
import torch
from torch import nn


@PROPOSAL_GENERATOR_REGISTRY.register()
class SAPRPN(RPN):
    '''
    difference between normal rpn and this is that output the cls logit in forward function and configure post_nms_topk for target domain
    '''
    @configurable
    def __init__(self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
        post_nms_topk_for_target_domain: int = 512
    ):
        super().__init__(in_features=in_features, head=head, anchor_generator=anchor_generator,\
            anchor_matcher=anchor_matcher, box2box_transform=box2box_transform, batch_size_per_image=batch_size_per_image,\
            positive_fraction=positive_fraction, pre_nms_topk=pre_nms_topk, post_nms_topk=post_nms_topk, nms_thresh=nms_thresh,\
            min_box_size=min_box_size, anchor_boundary_thresh=anchor_boundary_thresh, loss_weight=loss_weight,\
            box_reg_loss_type=box_reg_loss_type, smooth_l1_beta=smooth_l1_beta,\
        )
        self.post_nms_topk_for_target_domain = post_nms_topk_for_target_domain

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)
        ret["post_nms_topk_for_target_domain"] = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        output_pred_objectness_logits = pred_objectness_logits
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        is_target_domain = self.training and gt_instances is None
        if self.training and not is_target_domain:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes, is_target_domain
        )
        return proposals, losses, output_pred_objectness_logits

    def predict_proposals(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            image_sizes: List[Tuple[int, int]],
            is_target_domain: bool,
        ):
            """
            Decode all the predicted box regression deltas to proposals. Find the top proposals
            by applying NMS and removing boxes that are too small.

            Returns:
                proposals (list[Instances]): list of N Instances. The i-th Instances
                    stores post_nms_topk object proposals for image i, sorted by their
                    objectness score in descending order.
            """
            # The proposals are treated as fixed for joint training with roi heads.
            # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
            # are also network responses.

            # difference between this and default is post_nms_topk's setting, when features come from target domain, use another setting.
            # https://isrc.iscas.ac.cn/gitlab/research/domain-adaption/-/blob/master/detection/modeling/rpn.py#L89 
            if is_target_domain:
                post_nms_topk = self.post_nms_topk_for_target_domain
            else:
                post_nms_topk=self.post_nms_topk[self.training]
            with torch.no_grad():
                pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
                return find_top_rpn_proposals(
                    pred_proposals,
                    pred_objectness_logits,
                    image_sizes,
                    self.nms_thresh,
                    self.pre_nms_topk[self.training],
                    post_nms_topk,
                    self.min_box_size,
                    self.training,
                )