import json
from spider.model import SPIDER, SPIDER_MAST3R, SPIDER_FMwarp
import torch
import numpy as np
import random
from argparse import ArgumentParser
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import symmetric_inference as mast3r_symmetric_inference
from mast3r.inference import symmetric_inference_upsample as mast3r_symmetric_inference_upsample
from mast3r.cloud_opt.sparse_ga import extract_correspondences
from mast3r.inference import coarse_to_fine

from spider.utils.utils import match, match_upsample, sample_symmetric, to_pixel_coordinates, _to_pixel_coordinates, tensor_to_pil, fmwarp_match, match_keypoints
from spider.utils.image import load_original_images, resize_image_with_intrinsics
from spider.inference import symmetric_inference as spider_symmetric_inference
from spider.inference import symmetric_inference_upsample as spider_symmetric_inference_upsample
from spider.inference import fmwarp_symmetric_inference as spiderfmwarp_symmetric_inference
from spider.inference import spider_mast3r_symmetric_inference, spider_mast3r_symmetric_inference_upsample
import spider.utils.path_to_dust3r #noqa
from dust3r.utils.device import collate_with_cat
from matplotlib import pyplot as pl
import torch.nn.functional as F
import pdb
import sys
import os
import cv2
from PIL import Image

from tqdm import tqdm


def mast3r_match_path(save_dir, im_A_path, im_B_path, mast3r_model, device = 'cuda', coarse_size=512, fine_size=None, name='mast3r'):
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
    if fine_size == coarse_size or fine_size is None:
        imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
        view1, view2 = imgs_coarse
        view1, view2 = collate_with_cat([(view1, view2)])
        res = mast3r_symmetric_inference(mast3r_model, view1, view2, 'cuda')
        descs = [r['desc'][0] for r in res]
        qonfs = [r['desc_conf'][0] for r in res]  
        # perform reciprocal matching
        corres = extract_correspondences(descs, qonfs, device='cuda', subsample=16)
        mast3r_kpts1, mast3r_kpts2, mast3r_mconf = corres                                 
    else:
        imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
        imgs_fine, _ = resize_image_with_intrinsics(imgs_ori, size=fine_size, intrinsics=None, verbose=False)

        view1_coarse, view2_coarse = imgs_coarse
        view1_coarse, view2_coarse = collate_with_cat([(view1_coarse, view2_coarse)])
        view1, view2 = imgs_fine
        view1, view2 = collate_with_cat([(view1, view2)])
        
        mast3r_kpts1, mast3r_kpts2, mast3r_mconf = mast3r_symmetric_inference_upsample(mast3r_model, view1_coarse, view2_coarse, imgs_fine, 'cuda')
    mast3r_kpts1, mast3r_kpts2, mast3r_mconf = mast3r_kpts1.cpu().numpy(), mast3r_kpts2.cpu().numpy(), mast3r_mconf.cpu().numpy()
    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'][0] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    # img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img = np.concatenate((img0, img1), axis=1)
    # pl.figure()
    # pl.imshow(img)
    # pl.axis('off')  
    # pl.savefig(os.path.join(save_dir, 'raw.png'), dpi=300, bbox_inches='tight')
    # pl.close()
    
    # Draw matches
    # num_matches = len(mast3r_kpts1)
    # print(num_matches)
    def draw_matches(save_dir, kpts1, kpts2, mconf, name='mast3r_matches.png', n_viz = 400):
        valid_matches_im0 = (kpts1[:, 0] >= 3) & (kpts1[:, 0] < int(W0) - 3) & (
            kpts1[:, 1] >= 3) & (kpts1[:, 1] < int(H0) - 3)

        valid_matches_im1 = (kpts2[:, 0] >= 3) & (kpts2[:, 0] < int(W1) - 3) & (
            kpts2[:, 1] >= 3) & (kpts2[:, 1] < int(H1) - 3)
        valid_matches_im = mconf > 0.2
        valid_matches = valid_matches_im0 & valid_matches_im1 & valid_matches_im
        matches_im0, matches_im1 = kpts1[valid_matches], kpts2[valid_matches]
        matches_conf = mconf[valid_matches]
        num_matches = len(matches_im0)
        # print(num_matches)
        return num_matches

        # if num_matches > n_viz:
        #     # pick n_viz matches
        #     sorted_idx = np.argsort(-matches_conf)
            
        #     topk_idx = sorted_idx[:n_viz]
        #     viz_matches_im0 = matches_im0[topk_idx]
        #     viz_matches_im1 = matches_im1[topk_idx]
        #     viz_conf        = matches_conf[topk_idx]
        # else:
        #     n_viz = num_matches
        # viz_matches_im0 = matches_im0
        # viz_matches_im1 = matches_im1
        # pl.figure()
        # pl.imshow(img)
        # cmap = pl.get_cmap('jet')
        # for i in range(num_matches):
        #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        #     # pl.plot([x0, x1 + W0], [y0, y1], color=cmap(i / (n_viz - 1)), scalex=False, scaley=False, linewidth=0.3)
        #     pl.scatter(x0, y0, color=cmap(i / (num_matches - 1)), s=0.1)
        #     pl.scatter(x1 + W0, y1, color=cmap(i / (num_matches - 1)), s=0.1)
        # # pl.show(block=True)
        # pl.savefig(os.path.join(save_dir, f'{name}_matches-dot.png'), dpi=300, bbox_inches='tight')
        # pl.close()

        # pl.figure()
        # pl.imshow(img)
        # cmap = pl.get_cmap('jet')
        # for i in range(num_matches):
        #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        #     pl.plot([x0, x1 + W0], [y0, y1], color=cmap(i / (num_matches - 1)), scalex=False, scaley=False, linewidth=0.3)
        #     # pl.scatter(x0, y0, color=cmap(i / (n_viz - 1)), s=2)
        #     # pl.scatter(x1 + W0, y1, color=cmap(i / (n_viz - 1)), s=2)
        # # pl.show(block=True)
        # pl.savefig(os.path.join(save_dir, f'{name}_matches.png'), dpi=300, bbox_inches='tight')
        # pl.close()

    num_matches = draw_matches(save_dir, mast3r_kpts1, mast3r_kpts2, mast3r_mconf, name=name)
    return num_matches

if __name__ == '__main__':
    device = "cuda"
    # mast3r_model = AsymmetricMASt3R.from_pretrained("/cis/home/zshao14/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").to(device)
    mast3r_model = AsymmetricMASt3R.from_pretrained("/cis/home/zshao14/Downloads/spider/model_weights/aerial-mast3r.pth").to(device)
    pairs = np.load('/cis/net/r24a/data/zshao/data/doppelgangers/pairs_metadata/doppelgangers/pairs_metadata/test_pairs.npy', allow_pickle = True)
    save_dir = '/cis/home/zshao14/Documents/dg'
    data_dir = '/cis/net/r24a/data/zshao/data/doppelgangers/test_set/test_set'
    # pairs = np.load('/cis/net/r24a/data/zshao/data/doppelgangers/pairs_metadata/doppelgangers/pairs_metadata/train_pairs_megadepth.npy', allow_pickle = True)
    # save_dir = '/cis/home/zshao14/Documents/dg/train_megadepth'
    # data_dir = '/cis/net/r24a/data/zshao/data/doppelgangers/doppelgangers/images/train_megadepth'
    tot_0 = []
    tot_1 = []
    print(len(pairs))
    for idx, pair in enumerate(tqdm(pairs)):
        # if idx % 400 == 0: 
        im_A_path = os.path.join(data_dir, pair[0])
        im_B_path = os.path.join(data_dir, pair[1])
        name = pair[0].split('/')[0] + str(idx) + '_label' + str(pair[2]) + '_num' + str(pair[3])
        # print(name)
        save_path = os.path.join(save_dir, name)
        os.makedirs(save_path, exist_ok=True)
        try:
            with torch.no_grad():
                num_matches = mast3r_match_path(save_path, im_A_path, im_B_path, mast3r_model, device, coarse_size=512, fine_size=512, name='aerial-mast3r512')
            if pair[2] == 0:
                tot_0.append(num_matches)
            else:
                tot_1.append(num_matches)
        except Exception as e:
            print(f"Error processing {save_path}: {e}")
            continue
    print('Model: aerialmast3r512')
    print(f'Num Doppel {len(tot_0)} pairs, Average Matches: {np.mean(tot_0)}')
    print(f'Num Normal {len(tot_1)} pairs, Average Matches: {np.mean(tot_1)}')
