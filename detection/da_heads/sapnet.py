import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from .build import DA_HEAD_REGISTRY
from ..layers import GradientScalarLayer, style_pool2d

@DA_HEAD_REGISTRY.register()
class SAPNet(nn.Module):

    @configurable
    def __init__(self, *, num_anchors, in_channels, embedding_kernel_size=3,
            embedding_norm=True, embedding_dropout=True, func_name='cross_entropy',
            focal_loss_gamma=5, pool_type='avg', window_strides=None,
            window_sizes=(3, 9, 15, 21, -1)
        ):
        super().__init__()

        self.in_channels = in_channels
        self.embedding_kernel_size = embedding_kernel_size
        self.embedding_norm = embedding_norm
        self.embedding_dropout = embedding_dropout
        self.num_windows = len(window_sizes)
        self.num_anchors = num_anchors
        self.window_sizes = window_sizes
        if window_strides is None:
            self.window_strides = [None] * len(window_sizes)
        else:
            assert len(window_strides) == len(window_sizes), 'window_strides and window_sizes should has same len'
            self.window_strides = window_strides

        if pool_type == 'avg':
            channel_multiply = 1
            pool_func = F.avg_pool2d
        elif pool_type == 'max':
            channel_multiply = 1
            pool_func = F.max_pool2d
        elif pool_type == 'style':
            channel_multiply = 2
            pool_func = style_pool2d
        else:
            raise ValueError
        self.pool_type = pool_type
        self.pool_func = pool_func

        if func_name == 'cross_entropy':
            num_domain_classes = 2
            loss_func = F.cross_entropy

        else:
            raise ValueError
        self.focal_loss_gamma = focal_loss_gamma
        self.func_name = func_name
        self.loss_func = loss_func
        self.num_domain_classes = num_domain_classes

        NormModule = nn.BatchNorm2d if embedding_norm else nn.Identity
        DropoutModule = nn.Dropout if embedding_dropout else nn.Identity

        self.grl = GradientScalarLayer(-1.0)
        padding = (embedding_kernel_size - 1) // 2
        bias = not embedding_norm
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(in_channels),
            nn.ReLU(True),
            DropoutModule(),

            nn.Conv2d(in_channels, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(256),
            nn.ReLU(True),
            DropoutModule(),

            nn.Conv2d(256, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(256),
            nn.ReLU(True),
            DropoutModule(),
        )

        self.shared_semantic = nn.Sequential(
            nn.Conv2d(in_channels + 3 * self.num_anchors, in_channels, kernel_size=embedding_kernel_size, stride=1, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding),
            nn.ReLU(True),
        )

        self.semantic_list = nn.ModuleList()

        self.inter_channels = 128
        for _ in range(self.num_windows):
            self.semantic_list += [
                nn.Sequential(
                    nn.Conv2d(256, 128, 1),
                    nn.ReLU(True),
                    nn.Conv2d(128, 128, 1),
                    nn.ReLU(True),
                    nn.Conv2d(128, 1, 1),
                )
            ]

        self.fc = nn.Sequential(
            nn.Conv2d(256 * channel_multiply, 128, 1, bias=False),
            # NormModule(128), #
            nn.ReLU(inplace=True),
        )

        self.split_fc = nn.Sequential(
            nn.Conv2d(128, self.num_windows * 256 * channel_multiply, 1, bias=False),
        )

        self.predictor = nn.Linear(256 * channel_multiply, num_domain_classes)


    def forward(self, feature, rpn_logits, input_domain):
        '''
        feature: come from backbone [N, c, h, w]
        rpn_logits: come from rpn (anchors), list[tensor], [N, c, h, w] but in different size 
        '''

        feature = self.grl(feature)
        rpn_logits_ = []
        for  r in rpn_logits:
            r = self.grl(r)
            if feature.shape != r.shape:
                rpn_logits_.append(F.interpolate(r, size=(feature.size(2), feature.size(3)), mode='bilinear', align_corners=True))
            else:
                rpn_logits_.append(r)

        semantic_map = torch.cat([feature, *rpn_logits_], dim=1)
        semantic_map = self.shared_semantic(semantic_map)
        feature = self.embedding(feature)
        N, C, H, W = feature.shape

        pyramid_features = []
        for i, k in enumerate(self.window_sizes):
            if k == -1:
                x = self.pool_func(feature, kernel_size=(H, W))
            elif k == 1:
                x = feature
            else:
                stride = self.window_strides[i]
                if stride is None:
                    stride = 1  # default
                x = self.pool_func(feature, kernel_size=k, stride=stride)
            _, _, h, w = x.shape
            semantic_map_per_level = F.interpolate(semantic_map, size=(h, w), mode='bilinear', align_corners=True)
            domain_logits = self.semantic_list[i](semantic_map_per_level)

            w_spatial = domain_logits.view(N, -1)
            w_spatial = F.softmax(w_spatial, dim=1)
            w_spatial = w_spatial.view(N, 1, h, w)
            x = torch.sum(x * w_spatial, dim=(2, 3), keepdim=True)
            pyramid_features.append(x)

        fuse = sum(pyramid_features)  # [N, 256, 1, 1]
        merge = self.fc(fuse)  # [N, 128, 1, 1]
        split = self.split_fc(merge)  # [N, num_windows * 256, 1, 1]
        split = split.view(N, self.num_windows, -1, 1, 1)

        w = F.softmax(split, dim=1)
        w = torch.unbind(w, dim=1)  # List[N, 256, 1, 1]

        pyramid_features = list(map(lambda x, y: x * y, pyramid_features, w))
        final_features = sum(pyramid_features)
        del rpn_logits
        # del pyramid_features, w, split, merge, fuse, feature, rpn_logits
        final_features = final_features.view(N, -1)

        logits = self.predictor(final_features)
        if input_domain == 'source':
            domain_loss = self.loss_func(logits, torch.zeros(logits.size(0), dtype=torch.long, device=logits.device))
            return {'loss_sap_source_domain': domain_loss}
        elif input_domain == 'target':
            domain_loss = self.loss_func(logits, torch.ones(logits.size(0), dtype=torch.long, device=logits.device))
            return {'loss_sap_target_domain': domain_loss}


    @classmethod
    def from_config(cls, cfg):
        return {
            'num_anchors': cfg.MODEL.DA_HEAD.NUM_ANCHOR_IN_IMG,
            'in_channels': cfg.MODEL.DA_HEAD.IN_CHANNELS,
            'embedding_kernel_size': cfg.MODEL.DA_HEAD.EMBEDDING_KERNEL_SIZE,
            'embedding_norm': cfg.MODEL.DA_HEAD.EMBEDDING_NORM,
            'embedding_dropout': cfg.MODEL.DA_HEAD.EMBEDDING_DROPOUT,
            'func_name': cfg.MODEL.DA_HEAD.FUNC_NAME,
            'focal_loss_gamma': cfg.MODEL.DA_HEAD.FOCAL_LOSS_GAMMA,
            'pool_type': cfg.MODEL.DA_HEAD.POOL_TYPE,
            'window_strides': cfg.MODEL.DA_HEAD.WINDOW_STRIDES,
            'window_sizes': cfg.MODEL.DA_HEAD.WINDOW_SIZES,
        }

    # def __repr__(self):
    #     attrs = {
    #         'in_channels': self.in_channels,
    #         'embedding_kernel_size': self.embedding_kernel_size,
    #         'embedding_norm': self.embedding_norm,
    #         'embedding_dropout': self.embedding_dropout,
    #         'num_domain_classes': self.num_domain_classes,
    #         'func_name': self.func_name,
    #         'focal_loss_gamma': self.focal_loss_gamma,
    #         'pool_type': self.pool_type,
    #         'loss_weight': self.loss_weight,
    #         'window_strides': self.window_strides,
    #         'window_sizes': self.window_sizes,
    #     }
    #     table = AsciiTable(list(zip(attrs.keys(), attrs.values())))
    #     table.inner_heading_row_border = False
    #     return self.__class__.__name__ + '\n' + table.table

def build_da_heads(cfg):
    return SAPNet(cfg)