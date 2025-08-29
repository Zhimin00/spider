import json
from spider.model import SPIDER, SPIDER_MAST3R
import torch
import numpy as np
import random
from argparse import ArgumentParser
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import symmetric_inference as mast3r_symmetric_inference
from mast3r.inference import symmetric_inference_upsample as mast3r_symmetric_inference_upsample
from mast3r.cloud_opt.sparse_ga import extract_correspondences
from mast3r.inference import coarse_to_fine

from spider.utils.utils import match, match_upsample, sample_symmetric, to_pixel_coordinates, tensor_to_pil
from spider.utils.image import load_original_images, resize_image_with_intrinsics
from spider.inference import symmetric_inference as spider_symmetric_inference
from spider.inference import symmetric_inference_upsample as spider_symmetric_inference_upsample
from spider.inference import spider_mast3r_symmetric_inference, spider_mast3r_symmetric_inference_upsample
import spider.utils.path_to_dust3r #noqa
from dust3r.utils.device import collate_with_cat
from matplotlib import pyplot as pl
import os
import torch.nn.functional as F
import pdb

save_dir = '/cis/home/zshao14/Documents/BLH0001'
os.makedirs(save_dir, exist_ok=True)
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
        spider_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000, sample_mode = "balanced")
        spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, h1, w1, h2, w2)

        # mast3r inference
        mast3r_kpts1, mast3r_kpts2, mast3r_mconf = mast3r_symmetric_inference_upsample(mast3r_model, view1_coarse, view2_coarse, imgs_fine, 'cuda')

    # im2_transfer_rgb = F.grid_sample(
    #                         view2['img'][0][None].to(device), warp0[:,:, 2:][None], mode="bilinear", align_corners=False
    #                         )[0] ###H1, W1
    # im1_transfer_rgb = F.grid_sample(
    #                         view1['img'][0][None].to(device), warp1[:, :, :2][None], mode="bilinear", align_corners=False
    #                         )[0] ###H2, W2
    # white_im1 = torch.ones((h1, w1)).to(device)
    # white_im2 = torch.ones((h2, w2)).to(device)
    # pdb.set_trace()
    # vis_im1 = certainty0 * im2_transfer_rgb + (1 - certainty0) * white_im1
    # vis_im2 = certainty1 * im1_transfer_rgb + (1 - certainty1) * white_im2
    # tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp_im1.jpg'))
    # tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp_im2.jpg'))
    
    kpts1 = torch.cat((spider_kpts1, mast3r_kpts1), dim=0)
    kpts2 = torch.cat((spider_kpts2, mast3r_kpts2), dim=0)
    mconf = torch.cat((spider_mconf, mast3r_mconf), dim=0)
    return view1, view2, kpts1, kpts2, mconf, mast3r_kpts1, mast3r_kpts2, mast3r_mconf, spider_kpts1, spider_kpts2, spider_mconf
    # return view1, view2, spider_kpts1, spider_kpts2, spider_mconf

# def spider_mast3r_match_path(im_A_path, im_B_path, spider_mast3r_model, device = 'cuda', coarse_size=512, fine_size=None):
#     imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
#     if fine_size == coarse_size or fine_size is None:
#         imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
#         view1, view2 = imgs_coarse
#         view1, view2 = collate_with_cat([(view1, view2)])
        
#         # spider inference
#         corresps12, corresps21 = spider_symmetric_inference(spider_model, view1, view2, device)
#         warp0, certainty0 = match(corresps12)
#         warp1, certainty1 = match(corresps21, inverse=True)  
#         h1, w1 = imgs_coarse[0]['true_shape'][0]
#         h2, w2 = imgs_coarse[1]['true_shape'][0]
#         spider_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000)
#         spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, h1, w1, h2, w2)
        
#         # aerial-mast3r inference
#         res = mast3r_symmetric_inference(mast3r_model, view1, view2, 'cuda')
#         descs = [r['desc'][0] for r in res]
#         qonfs = [r['desc_conf'][0] for r in res]  
#         # perform reciprocal matching
#         corres = extract_correspondences(descs, qonfs, device='cuda', subsample=16)
#         mast3r_kpts1, mast3r_kpts2, mast3r_mconf = corres                                      
 
#     else:
#         imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
#         imgs_fine, _ = resize_image_with_intrinsics(imgs_ori, size=fine_size, intrinsics=None, verbose=False)

#         view1_coarse, view2_coarse = imgs_coarse
#         view1_coarse, view2_coarse = collate_with_cat([(view1_coarse, view2_coarse)])
#         view1, view2 = imgs_fine
#         view1, view2 = collate_with_cat([(view1, view2)])
        
#         # spider inference
#         low_corresps12, corresps12, low_corresps21, corresps21, res = spider_mast3r_symmetric_inference_upsample(spider_mast3r_model, view1_coarse, view2_coarse, view1, view2, device)
        
#         h1_coarse, w1_coarse = view1_coarse['true_shape'][0]
#         h2_coarse, w2_coarse = view2_coarse['true_shape'][0]
#         h1, w1 = view1['true_shape'][0]
#         h2, w2 = view2['true_shape'][0]

#         warp0, certainty0 = match_upsample(corresps12, low_corresps12)
#         warp1, certainty1 = match_upsample(corresps21, low_corresps21, inverse=True)
#         spider_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000)
#         spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, h1, w1, h2, w2)

#         descs = [r['desc'][0] for r in res]
#         qonfs = [r['desc_conf'][0] for r in res]  
#         # perform reciprocal matching
#         corres = extract_correspondences(descs, qonfs, device='cuda', subsample=8)
#         pts1, pts2, mconf = corres
        
#         kpts1 = (torch.stack( (
#                     (w1 / w1_coarse) * (pts1[..., 0]),
#                     (h1 / h1_coarse) * (pts1[..., 1]),
#                 ), axis=-1,))
#         kpts2 = ( torch.stack(  (
#                     (w2 / w2_coarse) * (pts2[..., 0]),
#                     (h2 / h2_coarse) * (pts2[..., 1]),
#                 ), axis=-1, )  )
                                                
#         mast3r_kpts1, mast3r_kpts2, mast3r_mconf = coarse_to_fine(h1, w1, h2, w2, imgs_fine, kpts1, kpts2, mconf, spider_mast3r_model, 'cuda')

    
#     kpts1 = torch.cat((spider_kpts1, mast3r_kpts1), dim=0)
#     kpts2 = torch.cat((spider_kpts2, mast3r_kpts2), dim=0)
#     mconf = torch.cat((spider_mconf, mast3r_mconf), dim=0)
#     return view1, view2, kpts1, kpts2, mconf

device = "cuda"
# spider_mast3r_model = SPIDER.from_pretrained('/cis/home/zshao14/Downloads/spider/spider_mast3r.pth').to(device)
spider_model = SPIDER.from_pretrained("/cis/home/zshao14/Downloads/spider/model_weights/spider.pth").to(device)
mast3r_model = AsymmetricMASt3R.from_pretrained("/cis/home/zshao14/Downloads/spider/model_weights/aerial-mast3r.pth").to(device)
# spider_model = SPIDER.from_pretrained("/cis/home/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth").to(device)
# mast3r_model = AsymmetricMASt3R.from_pretrained("/cis/home/zshao14/checkpoints/checkpoint-aerial-mast3r.pth").to(device)

with torch.no_grad():
    # im_A_path = '/cis/home/zshao14/Downloads/spider/assets/sacre_coeur/sacre_coeur_A.jpg'
    # im_B_path = '/cis/home/zshao14/Downloads/spider/assets/sacre_coeur/sacre_coeur_B.jpg'
    # im_A_path = '/cis/home/zshao14/Documents/M07_5/image_000001.jpg'
    # im_B_path = '/cis/home/zshao14/Documents/M07_5/image_000050.jpg'
    im_A_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/BLH0001/input/images/image_000075.JPG'
    im_B_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/BLH0001/input/images/image_000003.JPG'
    view1, view2, all_kpts1, all_kpts2, all_mconf, mast3r_kpts1, mast3r_kpts2, mast3r_mconf, spider_kpts1, spider_kpts2, spider_mconf = spider_mast3r_match_path(im_A_path, im_B_path, spider_model, mast3r_model, device, coarse_size=512, fine_size=1600)
    # view1, view2, kpts1, kpts2, mconf = spider_mast3r_match_path(im_A_path, im_B_path, spider_mast3r_model, device, coarse_size=512, fine_size=1600)
    all_kpts1, all_kpts2, all_mconf = all_kpts1.cpu().numpy(), all_kpts2.cpu().numpy(), all_mconf.cpu().numpy()
    spider_kpts1, spider_kpts2, spider_mconf = spider_kpts1.cpu().numpy(), spider_kpts2.cpu().numpy(), spider_mconf.cpu().numpy()
    mast3r_kpts1, mast3r_kpts2, mast3r_mconf = mast3r_kpts1.cpu().numpy(), mast3r_kpts2.cpu().numpy(), mast3r_mconf.cpu().numpy()

    
    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'][0] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    pl.axis('off')  
    pl.savefig(os.path.join(save_dir, 'raw.png'), dpi=300, bbox_inches='tight')
    pl.close()
    def draw_matches(kpts1, kpts2, mconf, name='mast3r_matches.png', n_viz = 400):
        valid_matches_im0 = (kpts1[:, 0] >= 3) & (kpts1[:, 0] < int(W0) - 3) & (
            kpts1[:, 1] >= 3) & (kpts1[:, 1] < int(H0) - 3)

        valid_matches_im1 = (kpts2[:, 0] >= 3) & (kpts2[:, 0] < int(W1) - 3) & (
            kpts2[:, 1] >= 3) & (kpts2[:, 1] < int(H1) - 3)
        valid_matches_im = mconf > 0.2
        valid_matches = valid_matches_im0 & valid_matches_im1 & valid_matches_im
        matches_im0, matches_im1 = kpts1[valid_matches], kpts2[valid_matches]
        matches_conf = mconf[valid_matches]

        # pick n_viz matches
        sorted_idx = np.argsort(-matches_conf)
        
        topk_idx = sorted_idx[:n_viz]
        viz_matches_im0 = matches_im0[topk_idx]
        viz_matches_im1 = matches_im1[topk_idx]
        viz_conf        = matches_conf[topk_idx]
        pdb.set_trace()
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False, linewidth=0.3)
            
        # pl.show(block=True)
        pl.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches='tight')
        pl.close()

        # num_matches = len(matches_im0)
        # match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    
        # pl.figure()
        # pl.imshow(img)
        # cmap = pl.get_cmap('jet')
        # for i in range(n_viz):
        #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        #     pl.plot([x0, x1 + W0], [y0, y1], '-', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False, linewidth=0.3)
            
        # # pl.show(block=True)
        # pl.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches='tight')
        # pl.close()
    draw_matches(all_kpts1, all_kpts2, all_mconf, name='matches.png')
    draw_matches(mast3r_kpts1, mast3r_kpts2, mast3r_mconf, name='mast3r_matches.png')
    draw_matches(spider_kpts1, spider_kpts2, spider_mconf, name='spider_matches.png')

    
                
