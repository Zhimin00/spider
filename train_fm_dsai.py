#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from spider.model import SPIDER_FM
from mast3r.losses import ConfMatchingLoss, MatchingLoss, APLoss, InfoNCE, MatchingLoss_Scale16, MatchingLoss_Scale8, MatchingLoss_Scale4, MatchingLoss_Scale2
from mast3r.datasets import Habitat, ARKitScenes, BlendedMVS, Co3d, MegaDepth, MegaDepth_all, Aerial_MegaDepth, ScanNetpp, StaticThings3D, Waymo, WildRGBD

import spider.training_fm
spider.training_fm.SPIDER_FM = SPIDER_FM
spider.training_fm.MatchingLoss = MatchingLoss
spider.training_fm.MatchingLoss_Scale16 = MatchingLoss_Scale16
spider.training_fm.MatchingLoss_Scale8 = MatchingLoss_Scale8
spider.training_fm.MatchingLoss_Scale4 = MatchingLoss_Scale4
spider.training_fm.MatchingLoss_Scale2 = MatchingLoss_Scale2
spider.training_fm.ConfMatchingLoss = ConfMatchingLoss
spider.training_fm.InfoNCE = InfoNCE
spider.training_fm.APLoss = APLoss

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

from dust3r.training import get_args_parser as dust3r_get_args_parser  # noqa
from spider.training_fm import train  # noqa


def get_args_parser():
    parser = dust3r_get_args_parser()
    # change defaults
    parser.prog = 'SPIDER_FM training'
    parser.set_defaults(model="SPIDER_FM(patch_embed_cls='ManyAR_PatchEmbed')")
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)