# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from mast3r.utils.collate import cat_collate, cat_collate_fn_map
from mast3r.utils.coarse_to_fine import select_pairs_of_crops
from mast3r.cloud_opt.sparse_ga import extract_correspondences
from mast3r.inference import crop
import spider.utils.path_to_dust3r #noqa
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf


from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r_visloc.datasets.utils import get_HW_resolution
import numpy as np

def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2



@torch.no_grad()
def twoheads_symmetric_inference(model, img1, img2, device):
    # combine all ref images into object-centric representation
    shape1 = img1['true_shape'].to(device, non_blocking=True)
    shape2 = img2['true_shape'].to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)
    # compute encoder only once
    feat1, feat2, pos1, pos2, cnn_feats1, cnn_feats2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2, cnn_feats1, cnn_feats2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        enc_output1, dec_output1 = dec1[0], dec1[-1]
        enc_output2, dec_output2 = dec2[0], dec2[-1]
        feat16_1 = torch.cat([enc_output1, dec_output1], dim=-1)
        feat16_2 = torch.cat([enc_output2, dec_output2], dim=-1)
        # cnn_feats1.append(feat16_1)
        # cnn_feats2.append(feat16_2)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(enabled=False):
                corresps = model._downstream_headwarp('warp', cnn_feats1 + [feat16_1], cnn_feats2 + [feat16_2], shape1, shape2)
        return corresps

    # decoder 1-2
    corresps12 = decoder(feat1, feat2, pos1, pos2, shape1, shape2, cnn_feats1, cnn_feats2)
    # decoder 2-1
    corresps21 = decoder(feat2, feat1, pos2, pos1, shape2, shape1, cnn_feats2, cnn_feats1)

    return (corresps12, corresps21)

@torch.no_grad()
def twoheads_symmetric_inference_upsample(model, img1_coarse, img2_coarse, img1, img2, device):
    # combine all ref images into object-centric representation
    low_corresps12, low_corresps21 = twoheads_symmetric_inference(model, img1_coarse, img2_coarse, device)
    shape1 = img1['true_shape'].to(device, non_blocking=True)
    shape2 = img2['true_shape'].to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)
    # compute encoder only once
    cnn_feats1, cnn_feats2 = model._encode_image_pairs_upsample(img1, img2, shape1, shape2)
    
    def decoder(shape1, shape2, cnn_feats1, cnn_feats2, finest_corresps=None):
        # cnn_feats1.append(feat16_1)
        # cnn_feats2.append(feat16_2)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(enabled=False):
                corresps = model._downstream_headwarp('warp', cnn_feats1, cnn_feats2, shape1, shape2, upsample=True, finest_corresps=finest_corresps)
        return corresps

    # decoder 1-2
    corresps12 = decoder(shape1, shape2, cnn_feats1, cnn_feats2, finest_corresps=low_corresps12[1])
    # decoder 2-1
    corresps21 = decoder(shape2, shape1, cnn_feats2, cnn_feats1, finest_corresps=low_corresps21[1])
    return (low_corresps12, corresps12, low_corresps21, corresps21)


@torch.no_grad()
def twoheads_concat_symmetric_inference(model, img1, img2, device):
    # combine all ref images into object-centric representation
    shape1 = img1['true_shape'].to(device, non_blocking=True)
    shape2 = img2['true_shape'].to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)
    # compute encoder only once
    feat1, feat2, pos1, pos2, cnn_feats1, cnn_feats2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2, cnn_feats1, cnn_feats2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        enc_output1, dec_output1 = dec1[0], dec1[-1]
        enc_output2, dec_output2 = dec2[0], dec2[-1]
        feat16_1 = torch.cat([enc_output1, dec_output1], dim=-1)
        feat16_2 = torch.cat([enc_output2, dec_output2], dim=-1)
        # cnn_feats1.append(feat16_1)
        # cnn_feats2.append(feat16_2)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(enabled=False):
                corresps = model._downstream_headwarp('warp', cnn_feats1 + [feat16_1], cnn_feats2 + [feat16_2], shape1, shape2)
                res1 = model._downstream_head(1, cnn_feats1 + [feat16_1], shape1, upsample = False, low_desc = None, low_certainty = None)
                res2 = model._downstream_head(2, cnn_feats2 + [feat16_2], shape2, upsample = False, low_desc = None, low_certainty = None)
        return corresps, res1, res2

    # decoder 1-2
    corresps12, res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2, cnn_feats1, cnn_feats2)
    # decoder 2-1
    corresps21, res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1, cnn_feats2, cnn_feats1)

    return (corresps12, corresps21), (res11, res21, res22, res12)

@torch.no_grad()
def twoheads_concat_symmetric_inference_upsample(model, img1_coarse, img2_coarse, img1, img2, device):
    # combine all ref images into object-centric representation
    low_corresps, res = twoheads_concat_symmetric_inference(model, img1_coarse, img2_coarse, device)
    low_corresps12, low_corresps21 = low_corresps
    shape1 = img1['true_shape'].to(device, non_blocking=True)
    shape2 = img2['true_shape'].to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)
    # compute encoder only once
    cnn_feats1, cnn_feats2 = model._encode_image_pairs_upsample(img1, img2, shape1, shape2)
    
    def decoder(shape1, shape2, cnn_feats1, cnn_feats2, finest_corresps=None):
        # cnn_feats1.append(feat16_1)
        # cnn_feats2.append(feat16_2)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(enabled=False):
                corresps = model._downstream_headwarp('warp', cnn_feats1, cnn_feats2, shape1, shape2, upsample=True, finest_corresps=finest_corresps)
        return corresps

    # decoder 1-2
    corresps12 = decoder(shape1, shape2, cnn_feats1, cnn_feats2, finest_corresps=low_corresps12[1])
    # decoder 2-1
    corresps21 = decoder(shape2, shape1, cnn_feats2, cnn_feats1, finest_corresps=low_corresps21[1])
    return (low_corresps12, corresps12, low_corresps21, corresps21), res



@torch.inference_mode()  # usually a bit faster than no_grad
def two_symmetric_inference_fast(model, img1, img2, device, use_amp=True):
    shape1 = img1['true_shape'].to(device, non_blocking=True)
    shape2 = img2['true_shape'].to(device, non_blocking=True)
    x1 = img1['img'].to(device, non_blocking=True)
    x2 = img2['img'].to(device, non_blocking=True)

    feat1, feat2, pos1, pos2, cw1, cw2, cf1, cf2 = model._encode_image_pairs(x1, x2, shape1, shape2)

    # Build batch for both directions: (1->2) and (2->1)
    # Assume batch assumed at dim0.
    featA = torch.cat([feat1, feat2], dim=0)   # query feats
    posA  = torch.cat([pos1,  pos2 ], dim=0)
    featB = torch.cat([feat2, feat1], dim=0)   # key feats
    posB  = torch.cat([pos2,  pos1 ], dim=0)

    shapeA = torch.cat([shape1, shape2], dim=0) if shape1.ndim > 0 else (shape1, shape2)
    shapeB = torch.cat([shape2, shape1], dim=0) if shape2.ndim > 0 else (shape2, shape1)

    def cat_list(L1, L2):
        # L1/L2 are lists of tensors with batch dim 0
        return [torch.cat([a, b], dim=0) for a, b in zip(L1, L2)]

    cwA = cat_list(cw1, cw2)
    cwB = cat_list(cw2, cw1)
    cfA = cat_list(cf1, cf2)
    cfB = cat_list(cf2, cf1)

    # Use AMP unless you know it hurts accuracy
    ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16)  # bf16 is safer than fp16
    with ctx:
        decA, decB = model._decoder(featA, posA, featB, posB)

        encA, outA = decA[0], decA[-1]
        encB, outB = decB[0], decB[-1]

        feat16_A = torch.cat([encA, outA], dim=-1)
        feat16_B = torch.cat([encB, outB], dim=-1)

        corresps = model._downstream_headwarp(
            'warp',
            cwA + [feat16_A],
            cwB + [feat16_B],
            shapeA, shapeB
        )

        # heads for both sides in one shot (still two calls because head(1) vs head(2))
        resA = model._downstream_head(1, cfA + [feat16_A], shapeA, upsample=False, low_desc=None, low_certainty=None)
        resB = model._downstream_head(2, cfB + [feat16_B], shapeB, upsample=False, low_desc=None, low_certainty=None)

    # Split back into (1->2) and (2->1)
    B = feat1.shape[0]
    corresps12, corresps21 = (corresps[0][:B], corresps[0][B:]), (corresps[1][:B], corresps[1][B:])
    # ^ adjust split logic to match your corresps structure

    # resA corresponds to "side-1 outputs" for batched directions; split similarly
    res11, res12 = resA[:B], resA[B:]
    res21, res22 = resB[:B], resB[B:]

    return (corresps12, corresps21), (res11, res21, res22, res12)


# @torch.no_grad()
@torch.inference_mode()
def two_symmetric_inference(model, img1, img2, device):
    shape1 = img1['true_shape'].to(device, non_blocking=True)
    shape2 = img2['true_shape'].to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)
    # compute encoder only once
    feat1, feat2, pos1, pos2, cnn_feats_warp1, cnn_feats_warp2, cnn_feats_fm1, cnn_feats_fm2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2, cnn_feats_warp1, cnn_feats_warp2, cnn_feats_fm1, cnn_feats_fm2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        enc_output1, dec_output1 = dec1[0], dec1[-1]
        enc_output2, dec_output2 = dec2[0], dec2[-1]
        feat16_1 = torch.cat([enc_output1, dec_output1], dim=-1)
        feat16_2 = torch.cat([enc_output2, dec_output2], dim=-1)
        # pdb.set_trace()
        # cnn_feats1.append(feat16_1)
        # cnn_feats2.append(feat16_2)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # with torch.cuda.amp.autocast(enabled=False):
            ctx = torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)  # bf16 is safer than fp16
            with ctx:
                corresps = model._downstream_headwarp('warp', cnn_feats_warp1 + [feat16_1], cnn_feats_warp2 + [feat16_2], shape1, shape2)
                res1 = model._downstream_head(1, cnn_feats_fm1 + [feat16_1], shape1, upsample = False, low_desc = None, low_certainty = None)
                res2 = model._downstream_head(2, cnn_feats_fm2 + [feat16_2], shape2, upsample = False, low_desc = None, low_certainty = None)
        return corresps, res1, res2

    # decoder 1-2
    corresps12, res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2, cnn_feats_warp1, cnn_feats_warp2, cnn_feats_fm1, cnn_feats_fm2)
    # decoder 2-1
    corresps21, res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1, cnn_feats_warp2, cnn_feats_warp1, cnn_feats_fm2, cnn_feats_fm1)

    return (corresps12, corresps21), (res11, res21, res22, res12)

@torch.no_grad()
def two_symmetric_inference_upsample(model, img1_coarse, img2_coarse, img1, img2, device):
    # combine all ref images into object-centric representation
    low_corresps, res = two_symmetric_inference(model, img1_coarse, img2_coarse, device)
    low_corresps12, low_corresps21 = low_corresps
    shape1 = img1['true_shape'].to(device, non_blocking=True)
    shape2 = img2['true_shape'].to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)
    # compute encoder only once
    cnn_feats1, cnn_feats2 = model._encode_image_pairs_upsample(img1, img2, shape1, shape2)
    
    def decoder(shape1, shape2, cnn_feats1, cnn_feats2, finest_corresps=None):
        # cnn_feats1.append(feat16_1)
        # cnn_feats2.append(feat16_2)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ctx = torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)  # bf16 is safer than fp16
            with ctx:#with torch.cuda.amp.autocast(enabled=False):
                corresps = model._downstream_headwarp('warp', cnn_feats1, cnn_feats2, shape1, shape2, upsample=True, finest_corresps=finest_corresps)
        return corresps

    # decoder 1-2
    corresps12 = decoder(shape1, shape2, cnn_feats1, cnn_feats2, finest_corresps=low_corresps12[1])
    # decoder 2-1
    corresps21 = decoder(shape2, shape1, cnn_feats2, cnn_feats1, finest_corresps=low_corresps21[1])
    return (low_corresps12, corresps12, low_corresps21, corresps21), res






def loss_of_one_batch_fm(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            pred1, pred2 = model(view1, view2)

            # loss is supposed to be symmetric
            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None
    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result


def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['pts3d', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            corresps = model(view1, view2)
            # loss is supposed to be symmetric
            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(view1, view2, corresps) if criterion is not None else None

    result = dict(view1=view1, view2=view2, corresps=corresps, loss=loss)
    return result[ret] if ret else result

@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []
    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result





def loss_of_one_batch_onlyfm(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            pred1, pred2 = model.onlyfm_forward(view1, view2)
            # loss is supposed to be symmetric
            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result

@torch.no_grad()
def crops_inference(pairs, model, device, batch_size=24, verbose=True):
    assert len(pairs) == 2, "Error, data should be a tuple of dicts containing the batch of image pairs"
    # Forward a possibly big bunch of data, by blocks of batch_size
    B = pairs[0]['img'].shape[0]
    if B < batch_size:
        return loss_of_one_batch_onlyfm(pairs, model, None, device=device, symmetrize_batch=False)
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {kk: pairs[di][kk][sel]
                             for kk in pairs[di].keys() if pairs[di][kk] is not None}  # copy chunk for forward
        preds.append(loss_of_one_batch_onlyfm(temp_data, model,
                                       None, device=device, symmetrize_batch=False))  # sequential forward
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)


def coarse_to_fine(h1, w1, h2, w2, imgs_large, kpts1, kpts2, mconf, model, device):
    crops1, crops2 = [], []
    to_orig1, to_orig2 = [], []
    query_resolution = get_HW_resolution(h1, w1, maxdim=512, patchsize=16)
    map_resolution = get_HW_resolution(h2, w2, maxdim=512, patchsize=16)
    img_large1, img_large2 = imgs_large[0]['img'][0].permute(1,2,0), imgs_large[1]['img'][0].permute(1,2,0)
    for crop_q, crop_b, pair_tag in select_pairs_of_crops(img_large1, img_large2, kpts1.cpu().numpy(),
                                                                    kpts2.cpu().numpy(),
                                                                    maxdim=512,
                                                                    overlap=0.5,
                                                                    forced_resolution=[query_resolution,
                                                                                        map_resolution]):
        c1, trf1 = crop(img_large1, crop_q)
        c2, trf2 = crop(img_large2, crop_b)
        crops1.append(c1)
        crops2.append(c2)
        to_orig1.append(trf1)
        to_orig2.append(trf2)
    if len(crops1) == 0 or len(crops2) == 0:
        return kpts1, kpts2, mconf
    else:
        crops1, crops2 = torch.stack(crops1), torch.stack(crops2)
        if len(crops1.shape) == 3:
            crops1, crops2 = crops1[None], crops2[None]
        to_orig1, to_orig2 = torch.stack(to_orig1), torch.stack(to_orig2)
        query_crop_view = dict(img=crops1.permute(0, 3, 1, 2),
                                instance=['1' for _ in range(crops1.shape[0])],
                                true_shape=torch.from_numpy(np.int32(query_resolution)).unsqueeze(0).repeat(crops1.shape[0], 1),
                                to_orig=to_orig1)
        map_crop_view = dict(img=crops2.permute(0, 3, 1, 2),
                                instance=['2' for _ in range(crops2.shape[0])],
                                true_shape=torch.from_numpy(np.int32(map_resolution)).unsqueeze(0).repeat(crops2.shape[0], 1),
                                to_orig=to_orig2)
        

        # Inference and Matching
        kpts1, kpts2, mconf = fine_matching(query_crop_view, map_crop_view, model, device)
    return kpts1, kpts2, mconf

def fine_matching(query_views, map_views, model, device, max_batch_size=48):
    output = crops_inference([query_views, map_views],
                             model, device, batch_size=max_batch_size, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    descs1 = pred1['desc'].clone()
    descs2 = pred2['desc'].clone()
    confs1 = pred1['desc_conf'].clone()
    confs2 = pred2['desc_conf'].clone()

    # Compute matches
    matches_im_map, matches_im_query, matches_confs = [], [], []
    for ppi, (pp1, pp2, cc11, cc21) in enumerate(zip(descs1, descs2, confs1, confs2)):
        conf_list_ppi = [cc11, cc21]
        matches_im_map_ppi, matches_im_query_ppi = fast_reciprocal_NNs(pp2, pp1, subsample_or_initxy1=8,
                                                                       **dict(device=device, dist='dot', block_size=2**13))
        matches_confs_ppi = torch.minimum(
            conf_list_ppi[1][matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]],
            conf_list_ppi[0][matches_im_query_ppi[:, 1], matches_im_query_ppi[:, 0]]
        )
        # inverse operation where we uncrop pixel coordinates
        device = map_views['to_orig'][ppi].device
        matches_im_map_ppi = torch.from_numpy(matches_im_map_ppi.copy()).float().to(device)
        device = query_views['to_orig'][ppi].device
        matches_im_query_ppi = torch.from_numpy(matches_im_query_ppi.copy()).float().to(device)
        matches_im_map_ppi = geotrf(map_views['to_orig'][ppi], matches_im_map_ppi, norm=True)
        matches_im_query_ppi = geotrf(query_views['to_orig'][ppi], matches_im_query_ppi, norm=True)
        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)
        matches_confs.append(matches_confs_ppi)

    matches_im_map = torch.cat(matches_im_map, dim=0)
    matches_im_query = torch.cat(matches_im_query, dim=0)
    matches_confs = torch.cat(matches_confs, dim=0)
    return matches_im_query, matches_im_map, matches_confs

def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d

