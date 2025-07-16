# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions for MASt3R
# --------------------------------------------------------
import os
import hashlib
import torch
import pdb

def mkdir_for(f):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    return f


def hash_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def interleave_list(list1, list2):
    assert len(list1) == len(list2), 'cnn features have different length (scales)'
    res1 = [torch.stack((list1[idx], list2[idx]), dim=1).flatten(0, 1) for idx in range(len(list1))]
    res2 = [torch.stack((list2[idx], list1[idx]), dim=1).flatten(0, 1) for idx in range(len(list1))] 
    return res1, res2


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}   #B, H, W, C

def transposed_corresps(dic):
    result = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            result[k] = transposed_corresps(v)
        else:
            result[k] = v.swapaxes(-1, -2)
    return result


def transpose_to_landscape_warp(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(cnn_feats1, cnn_feats2, true_shape1, true_shape2, upsample = False, finest_corresps=None):
        B = len(true_shape1)
        assert true_shape1[0:1].allclose(true_shape1), 'true_shape1 must be all identical'
        assert true_shape2[0:1].allclose(true_shape2), 'true_shape2 must be all identical'
        H1, W1 = true_shape1[0].cpu().tolist()
        H2, W2 = true_shape2[0].cpu().tolist()
        res = head(cnn_feats1, cnn_feats2, (H1, W1), (H2, W2), upsample=upsample, finest_corresps=finest_corresps)
        return res

    def wrapper_yes(cnn_feats1, cnn_feats2, true_shape1, true_shape2, upsample = False, finest_corresps=None):
        B = len(true_shape1)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape1.min()), int(true_shape1.max())

        height1, width1 = true_shape1.T
        height2, width2 = true_shape2.T
        is_land2land = (width1 >= height1) & (width2 >= height2)
        is_land2port = (width1 >= height1) & (width2 < height2)
        is_port2land = (width1 < height1) & (width2 >= height2)
        is_port2port = (width1 < height1) & (width2 < height2)

        # true_shape = true_shape.cpu()
        if is_land2land.all():
            return head(cnn_feats1, cnn_feats2, (H, W), (H, W), upsample=upsample, finest_corresps=finest_corresps)
        if is_land2port.all():
            return head(cnn_feats1, cnn_feats2, (H, W), (W, H), upsample=upsample, finest_corresps=finest_corresps)
        if is_port2land.all():
            return transposed_corresps(head(cnn_feats1, cnn_feats2, (W, H), (H, W), upsample=upsample, finest_corresps=finest_corresps))
        if is_port2port.all():
            return transposed_corresps(head(cnn_feats1, cnn_feats2, (W, H), (W, H), upsample=upsample, finest_corresps=finest_corresps))

        # batch is a mix of both portraint & landscape
        def cnnout1(ar): return [cnn_feat[ar] for cnn_feat in cnn_feats1]
        def cnnout2(ar): return [cnn_feat[ar] for cnn_feat in cnn_feats2]

        cases = [
            ("land2land", is_land2land, (H, W), (H, W), False),
            ("land2port", is_land2port, (H, W), (W, H), False),
            ("port2land", is_port2land, (W, H), (H, W), True),
            ("port2port", is_port2port, (W, H), (W, H), True),
        ]

        partial_results = {}

        for name, mask, shape1, shape2, transpose in cases:
            if mask.any():
                out1 = cnnout1(mask)
                out2 = cnnout2(mask)
                head_result = head(out1, out2, shape1, shape2)
                if transpose:
                    head_result = transposed_corresps(head_result)
                partial_results[name] = head_result
        # allocate and fill final result
        result = {}
        template = next(iter(partial_results.values()))

        for s, inner in template.items():
            result[s] = {}
            for k, sample_tensor in inner.items():
                shape = (B, *sample_tensor.shape[1:])
                x = sample_tensor.new_zeros(shape)

                # fill from each partial result
                for name, mask, *_ in cases:
                    if name in partial_results:
                        x[mask] = partial_results[name][s][k]

                result[s][k] = x
        return result

    return wrapper_yes if activate else wrapper_no
