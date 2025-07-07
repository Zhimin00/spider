#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from spider.model import SPIDER
import spider.training
spider.training.SPIDER = SPIDER

import spider.utils.path_to_dust3r  # noqa
from dust3r.training import get_args_parser as dust3r_get_args_parser  # noqa
from spider.training import train  # noqa


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
