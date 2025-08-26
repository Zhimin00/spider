import json
from spider.model import SPIDER
import torch
import numpy as np
import random
from argparse import ArgumentParser
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import symmetric_inference as mast3r_symmetric_inference
from mast3r.inference import symmetric_inference_upsample as mast3r_symmetric_inference_upsample
from mast3r.cloud_opt.sparse_ga import extract_correspondences

from spider.utils.utils import match, match_upsample, sample_symmetric, to_pixel_coordinates
from spider.utils.image import load_original_images, resize_image_with_intrinsics
from spider.inference import symmetric_inference as spider_symmetric_inference
from spider.inference import symmetric_inference_upsample as spider_symmetric_inference_upsample

import spider.utils.path_to_dust3r #noqa
from dust3r.utils.device import collate_with_cat

def spider_mast3r_match_path(im_A_path, im_B_path, spider_model, mast3r_model, device = 'cuda', coarse_size=512, fine_size=None, is_scannet=False):
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
    if fine_size == coarse_size or fine_size is None:
        imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
        view1, view2 = imgs_coarse
        view1, view2 = collate_with_cat([(view1, view2)])
        
        # spider inference
        corresps12, corresps21 = spider_symmetric_inference(spider_model, view1, view2, device)
        warp0, certainty0 = match(corresps12)
        warp1, certainty1 = match(corresps21, inverse=True)  
        h1, w1 = imgs_coarse[0]['true_shape'][0]
        h2, w2 = imgs_coarse[1]['true_shape'][0]
        spider_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000)
        spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, h1, w1, h2, w2)
        
        # aerial-mast3r inference
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
        
        # spider inference
        low_corresps12, corresps12, low_corresps21, corresps21 = spider_symmetric_inference_upsample(spider_model, view1_coarse, view2_coarse, view1, view2, device)
        warp0, certainty0 = match_upsample(corresps12, low_corresps12)
        warp1, certainty1 = match_upsample(corresps21, low_corresps21, inverse=True)
        h1, w1 = imgs_fine[0]['true_shape'][0]
        h2, w2 = imgs_fine[1]['true_shape'][0]
        spider_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000)
        spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, h1, w1, h2, w2)

        # mast3r inference
        mast3r_kpts1, mast3r_kpts2, mast3r_mconf = mast3r_symmetric_inference_upsample(mast3r_model, view1_coarse, view2_coarse, imgs_fine, 'cuda')

    kpts1 = torch.cat((spider_kpts1, mast3r_kpts1), dim=0)
    kpts2 = torch.cat((spider_kpts2, mast3r_kpts2), dim=0)
    mconf = torch.cat((spider_mconf, mast3r_mconf), dim=0)
    return kpts1, kpts2, mconf

device = "cuda"
spider_model = SPIDER.from_pretrained("/cis/home/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth").to(device)
mast3r_model = AsymmetricMASt3R.from_pretrained("/cis/home/zshao14/checkpoints/checkpoint-aerial-mast3r.pth").to(device)

with torch.no_grad():
    im_A_path = '/cis/home/zshao14/Downloads/spider/assets/sacre_coeur/sacre_coeur_A.jpg'
    im_B_path = '/cis/home/zshao14/Downloads/spider/assets/sacre_coeur/sacre_coeur_B.jpg'

    kpts1, kpts2, mconf = spider_mast3r_match_path(im_A_path, im_B_path, spider_model, mast3r_model, device, coarse_size=512, fine_size=1600)
    kpts1, kpts2, mconf = kpts1.cpu().numpy(), kpts2.cpu().numpy(), mconf.cpu().numpy()

    
                
