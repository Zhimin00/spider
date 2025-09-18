#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from spider.model import DINOv3_SPIDER
import spider.training
spider.training.DINOv3_SPIDER = DINOv3_SPIDER

from mast3r.losses import ConfMatchingLoss, MatchingLoss, WarpMatchingLoss, APLoss, InfoNCE, Regr3D, Regr3D_ScaleShiftInv
from mast3r.datasets import Habitat, ARKitScenes, BlendedMVS, Co3d, MegaDepth, MegaDepth_all, Aerial_MegaDepth, ScanNetpp, StaticThings3D, Waymo, WildRGBD


import spider.training_dinov3
spider.training_dinov3.DINOv3_SPIDER = DINOv3_SPIDER
spider.training_dinov3.MatchingLoss = MatchingLoss
spider.training_dinov3.WarpMatchingLoss = WarpMatchingLoss
spider.training_dinov3.ConfMatchingLoss = ConfMatchingLoss
spider.training_dinov3.InfoNCE = InfoNCE
spider.training_dinov3.APLoss = APLoss
spider.training_dinov3.Regr3D = Regr3D
spider.training_dinov3.Regr3D_ScaleShiftInv = Regr3D_ScaleShiftInv

import spider.utils.path_to_dust3r  # noqa
import dust3r.datasets
dust3r.datasets.Habitat = Habitat
dust3r.datasets.ARKitScenes = ARKitScenes
dust3r.datasets.BlendedMVS = BlendedMVS
dust3r.datasets.Co3d = Co3d
dust3r.datasets.MegaDepth = MegaDepth
dust3r.datasets.MegaDepth_all = MegaDepth_all
dust3r.datasets.Aerial_MegaDepth = Aerial_MegaDepth
dust3r.datasets.ScanNetpp = ScanNetpp
dust3r.datasets.StaticThings3D = StaticThings3D
dust3r.datasets.Waymo = Waymo
dust3r.datasets.WildRGBD = WildRGBD

from spider.training_dinov3 import get_args_parser as dust3r_get_args_parser  # noqa
from spider.training_dinov3 import train  # noqa


def get_args_parser():
    parser = dust3r_get_args_parser()
    # change defaults
    parser.prog = 'MASt3R training'
    parser.set_defaults(model="AsymmetricMASt3R(patch_embed_cls='ManyAR_PatchEmbed')")
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
