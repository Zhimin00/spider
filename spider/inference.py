# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
import spider.utils.path_to_dust3r #noqa
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf


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
