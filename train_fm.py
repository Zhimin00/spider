#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from spider.model import SPIDER_FM
from mast3r.losses import ConfMatchingLoss, MatchingLoss, APLoss, InfoNCE
from mast3r.datasets import Habitat, ARKitScenes, BlendedMVS, Co3d, MegaDepth, MegaDepth_all, Aerial_MegaDepth, ScanNetpp, StaticThings3D, Waymo, WildRGBD

import mast3r.utils.path_to_dust3r  # noqa
# add mast3r classes to dust3r imports
import dust3r.training
dust3r.training.SPIDER_FM = SPIDER_FM
dust3r.training.MatchingLoss = MatchingLoss
dust3r.training.ConfMatchingLoss = ConfMatchingLoss
dust3r.training.InfoNCE = InfoNCE
dust3r.training.APLoss = APLoss

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
from dust3r.training import train  # noqa


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