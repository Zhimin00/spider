#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from spider.model import SPIDER_twoheads
import spider.training
spider.training.SPIDER_twoheads = SPIDER_twoheads

from mast3r.losses import ConfMatchingLoss, ConfMatchingLoss3, MatchingLoss, WarpMatchingLoss, WarpMatchingLoss2, WarpMatchingLoss3, APLoss, InfoNCE, InfoNCE_weighted, MatchingLoss_Scale16, MatchingLoss_Scale8, MatchingLoss_Scale4, MatchingLoss_Scale2
from mast3r.datasets import Habitat, ARKitScenes, BlendedMVS, Co3d, MegaDepth, MegaDepth_all, Aerial_MegaDepth, ScanNetpp, StaticThings3D, Waymo, WildRGBD


import spider.training_twoheads
spider.training_twoheads.SPIDER_twoheads = SPIDER_twoheads
spider.training_twoheads.MatchingLoss = MatchingLoss
spider.training_twoheads.WarpMatchingLoss = WarpMatchingLoss
spider.training_twoheads.WarpMatchingLoss2 = WarpMatchingLoss2
spider.training_twoheads.WarpMatchingLoss3 = WarpMatchingLoss3
spider.training_twoheads.ConfMatchingLoss = ConfMatchingLoss
spider.training_twoheads.ConfMatchingLoss = ConfMatchingLoss3
spider.training_twoheads.InfoNCE = InfoNCE
spider.training_twoheads.InfoNCE_weighted = InfoNCE_weighted
spider.training_twoheads.APLoss = APLoss

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

from spider.training_twoheads import get_args_parser as dust3r_get_args_parser  # noqa
from spider.training_twoheads import train  # noqa


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
