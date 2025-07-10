#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from spider.model import SPIDER_POINTMAP

import spider.utils.path_to_dust3r  # noqa
import dust3r.training
dust3r.training.SPIDER_POINTMAP = SPIDER_POINTMAP

from dust3r.training import get_args_parser, train  # noqa


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
