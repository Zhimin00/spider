# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from mast3r.utils.collate import cat_collate, cat_collate_fn_map
import spider.utils.path_to_dust3r #noqa
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
import pdb
import gc
import time

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
def symmetric_inference(model, img1, img2, device):
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
                corresps = model._downstream_head(1, cnn_feats1 + [feat16_1], cnn_feats2 + [feat16_2], shape1, shape2)
        return corresps

    # decoder 1-2
    corresps12 = decoder(feat1, feat2, pos1, pos2, shape1, shape2, cnn_feats1, cnn_feats2)
    # decoder 2-1
    corresps21 = decoder(feat2, feat1, pos2, pos1, shape2, shape1, cnn_feats2, cnn_feats1)

    return (corresps12, corresps21)

@torch.no_grad()
def symmetric_inference_upsample(model, img1_coarse, img2_coarse, img1, img2, device):
    # combine all ref images into object-centric representation
    low_corresps12, low_corresps21 = symmetric_inference(model, img1_coarse, img2_coarse, device)
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
                corresps = model._downstream_head(1, cnn_feats1, cnn_feats2, shape1, shape2, upsample=True, finest_corresps=finest_corresps)
        return corresps

    # decoder 1-2
    corresps12 = decoder(shape1, shape2, cnn_feats1, cnn_feats2, finest_corresps=low_corresps12[1])
    # decoder 2-1
    corresps21 = decoder(shape2, shape1, cnn_feats2, cnn_feats1, finest_corresps=low_corresps21[1])
    # torch.cuda.empty_cache()
    # time.sleep(0.2)
    # time.sleep(0.01)

    return (low_corresps12, corresps12, low_corresps21, corresps21)

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

def loss_of_one_batch_upsample(batch, upsample_batch, model, criterion, device, use_amp=False, ret=None):
    ignore_keys = {'pts3d', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'}

    def move_to_device(view_list):
        for view in view_list:
            for k, v in view.items():
                if k not in ignore_keys:
                    view[k] = v.to(device, non_blocking=True)
        return view_list

    # Move low-res views to device
    view1, view2 = move_to_device(batch)
    # Forward pass (low-res)
    with torch.cuda.amp.autocast(enabled=use_amp):
        low_corresps = model(view1, view2)
        finest_corresps = low_corresps[1]

    # Move high-res views to device
    view1_up, view2_up = move_to_device(upsample_batch)

    # Forward match (high-res)
    with torch.cuda.amp.autocast(enabled=use_amp):
        corresps = model.match(view1_up, view2_up, finest_corresps)

    # Optional loss
    loss = None
    if criterion is not None:
        with torch.cuda.amp.autocast(enabled=False):  # loss in full precision
            loss = criterion(view1_up, view2_up, corresps)

    # Optionally clean up
    result = {
        'corresps': corresps,
        'loss': loss,
        'low_corresps': low_corresps
    }

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

@torch.no_grad()
def inference_upsample(pairs, upsample_pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs)) or not (check_if_same_size(upsample_pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch_upsample(collate_with_cat(pairs[i:i + batch_size]), collate_with_cat(upsample_pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))
        torch.cuda.empty_cache()
        time.sleep(0.01)

    result = collate_with_cat(result, lists=multiple_shapes)

    return result

@torch.no_grad()
def inference_cuda(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(res)

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


@torch.no_grad()
def inference_upsample_cuda(pairs, upsample_pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs)) or not (check_if_same_size(upsample_pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch_upsample(collate_with_cat(pairs[i:i + batch_size]), collate_with_cat(upsample_pairs[i:i + batch_size]), model, None, device)
        result.append(res)
        torch.cuda.empty_cache()
        time.sleep(0.1)

    result = collate_with_cat(result, lists=multiple_shapes)
    return result

@torch.no_grad()
def crops_inference(pairs, model, device, batch_size=48, verbose=True):
    assert len(pairs) == 2, "Error, data should be a tuple of dicts containing the batch of image pairs"
    # Forward a possibly big bunch of data, by blocks of batch_size
    B = pairs[0]['img'].shape[0]
    if B < batch_size:
        return loss_of_one_batch(pairs, model, None, device=device, symmetrize_batch=False)
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {kk: pairs[di][kk][sel]
                             for kk in pairs[di].keys() if pairs[di][kk] is not None}  # copy chunk for forward
        preds.append(loss_of_one_batch(temp_data, model,
                                       None, device=device, symmetrize_batch=False))  # sequential forward
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)

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
