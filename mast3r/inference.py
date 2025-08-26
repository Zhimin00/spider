import torch
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.coarse_to_fine import select_pairs_of_crops, crop_slice
from mast3r.utils.collate import cat_collate, cat_collate_fn_map
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.cloud_opt.sparse_ga import extract_correspondences
import mast3r.utils.path_to_dust3r # noqa
from dust3r_visloc.datasets.utils import get_HW_resolution
from dust3r.inference import loss_of_one_batch
from dust3r.utils.geometry import geotrf, find_reciprocal_matches, xy_grid
import numpy as np

def crop(img, crop):
    out_cropped_img = img.clone()
    to_orig = torch.eye(3, device=img.device)
    out_cropped_img = img[crop_slice(crop)]
    to_orig[:2, -1] = torch.tensor(crop[:2])

    return out_cropped_img, to_orig

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
    # Merge all preds
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)

@torch.no_grad()
def symmetric_inference(model, img1, img2, device):
    shape1 = img1['true_shape'].to(device, non_blocking=True)
    shape2 = img2['true_shape'].to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return (res11, res21, res22, res12)

@torch.no_grad()
def symmetric_inference_upsample(model, img1_coarse, img2_coarse, imgs_fine, device):
    res = symmetric_inference(model, img1_coarse, img2_coarse, 'cuda')
    descs = [r['desc'][0] for r in res]
    qonfs = [r['desc_conf'][0] for r in res]  
    # perform reciprocal matching
    corres = extract_correspondences(descs, qonfs, device='cuda', subsample=8)
    pts1, pts2, mconf = corres

    h1_coarse, w1_coarse = img1_coarse['true_shape'][0]
    h2_coarse, w2_coarse = img2_coarse['true_shape'][0]

    h1, w1 = imgs_fine[0]['true_shape'][0]
    h2, w2 = imgs_fine[1]['true_shape'][0]
    kpts1 = (
        torch.stack(
            (
                (w1 / w1_coarse) * (pts1[..., 0]),
                (h1 / h1_coarse) * (pts1[..., 1]),
            ),
            axis=-1,
        )
    )
    kpts2 = (
        torch.stack(
            (
                (w2 / w2_coarse) * (pts2[..., 0]),
                (h2 / h2_coarse) * (pts2[..., 1]),
            ),
            axis=-1,
        )
    )
                                            
    kpts1, kpts2, mconf = coarse_to_fine(h1, w1, h2, w2, imgs_fine, kpts1, kpts2, mconf, model, 'cuda')
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