import json
from spider.model import SPIDER, SPIDER_MAST3R, SPIDER_FMwarp
import torch
import numpy as np
import random
from argparse import ArgumentParser


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

hloc_jhu_path = os.path.abspath('/cis/net/r24a/data/zshao/jhu-registration/Hierarchical-Localization')
sys.path.insert(0, hloc_jhu_path)
from hloc import extract_features, extractors
from hloc.utils.base_model import dynamic_load

from DeDoDe import dedode_detector_L
save_dir = '/cis/home/zshao14/Documents/BLH0001_vis2'
dad_path = os.path.abspath('/cis/home/zshao14/Downloads/dad')
sys.path.insert(0, dad_path)
import dad as dad_detector


detector = dedode_detector_L(weights = None)
detector_dad = dad_detector.load_DaD()
feature_conf = extract_features.confs['superpoint_max']
Model = dynamic_load(extractors, feature_conf["model"]["name"])
sp_model = Model(feature_conf["model"]).eval().to('cpu')

os.makedirs(save_dir, exist_ok=True)


def match_single16(im_A_to_im_B, certainty, hs, ws, inverse=False, batched=False):
    """
    not batched:
        im_A_to_im_B:   2, h, w
        certainty:      1, h, w
        certainty_s16:  1, h//16, w//16
    batched:
        im_A_to_im_B:   B, 2, h, w
        certainty:      B, 1, h, w
        certainty_s16:  B, 1, h//16, w//16
    """
    if im_A_to_im_B.ndim == 3:
        assert batched == False
        im_A_to_im_B, certainty, certainty_s16 = im_A_to_im_B[None], certainty[None], certainty_s16[None]
    b, _, h, w = certainty.shape
    if not batched:
        assert b == 1

    im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
    certainty = F.interpolate(
        certainty, size=(hs, ws), align_corners=False, mode="bilinear"
    )
    pdb.set_trace()
    im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
    device = im_A_to_im_B.device
    # Create im_A meshgrid
    im_A_coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
            torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
        ),
        indexing='ij'
    )
    im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
    im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
    certainty = certainty.sigmoid()  # logits -> probs
    im_A_coords = im_A_coords.permute(0, 2, 3, 1)
    if (im_A_to_im_B.abs() > 1).any() and True:
        wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
        certainty[wrong[:, None]] = 0
    im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
    if inverse:
        warp = torch.cat((im_A_to_im_B, im_A_coords), dim=-1)
    else:
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
    if batched:
        return (
            warp, ### b, h, w, 4
            certainty[:,0] ### b, h, w
        )
    return (
            warp[0], ### h, w, 4
            certainty[0, 0] ### h, w
        )

def match16(corresps, attenuate_cert=False, inverse=False):
    finest_scale = 16
    im_A_to_im_Bs = corresps[finest_scale]["flow"] 
    certaintys = corresps[finest_scale]["certainty"]
    _, _, h, w = corresps[1]["certainty"].shape
    warp, certainty = match_single16(im_A_to_im_B=im_A_to_im_Bs, certainty=certaintys, hs=h, ws=w, inverse=inverse)
    return warp, certainty

def spider_match_path(im_A_path, im_B_path, spider_model, device = 'cuda', coarse_size=512, fine_size=None, name='spider'):
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
    
    imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
    view1, view2 = imgs_coarse
    view1, view2 = collate_with_cat([(view1, view2)])
    
    # spider inference
    corresps12, corresps21 = spider_symmetric_inference(spider_model, view1, view2, device)
    warp0, certainty0 = match16(corresps12)
    warp1, certainty1 = match16(corresps21, inverse=True)  
    h1, w1 = imgs_coarse[0]['true_shape'][0]
    h2, w2 = imgs_coarse[1]['true_shape'][0]
     
    img1, img2 = cv2.imread(im_A_path), cv2.imread(im_B_path)
    W_A, H_A = Image.open(im_A_path).size
    W_B, H_B = Image.open(im_B_path).size

    
    im1 = Image.open(im_A_path).resize((h1, w1))
    im2 = Image.open(im_B_path).resize((h2, w2))
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)
    im2_transfer_rgb = F.grid_sample(
                            x2[None], warp0[:,:, 2:][None], mode="bilinear", align_corners=False
                            )[0] ###H1, W1
    im1_transfer_rgb = F.grid_sample(
                            x1[None], warp1[:, :, :2][None], mode="bilinear", align_corners=False
                            )[0] ###H2, W2

    white_im1 = torch.ones((h1, w1)).to(device)
    white_im2 = torch.ones((h2, w2)).to(device)
    vis_im1 = certainty0 * im2_transfer_rgb + (1 - certainty0) * white_im1
    vis_im2 = certainty1 * im1_transfer_rgb + (1 - certainty1) * white_im2
    tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp16_im1.jpg'))
    tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp16_im2.jpg'))
    # return view1, view2, spider_kpts1, spider_kpts2, spider_mconf
 

device = "cuda"
spider_model = SPIDER.from_pretrained("/cis/home/zshao14/Downloads/spider/model_weights/spider.pth").to(device)
# spider_model = SPIDER_FMwarp.from_pretrained('/cis/home/zshao14/checkpoints/spider_mast3r_fmwarp16_0828_new/checkpoint-best.pth').to(device)
with torch.no_grad():
    # im_A_path = '/cis/home/zshao14/Downloads/spider/assets/sacre_coeur/sacre_coeur_A.jpg'
    # im_B_path = '/cis/home/zshao14/Downloads/spider/assets/sacre_coeur/sacre_coeur_B.jpg'
    # im_A_path = '/cis/home/zshao14/Documents/M07_5/image_000001.jpg'
    # im_B_path = '/cis/home/zshao14/Documents/M07_5/image_000050.jpg'
    im_A_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/BLH0001/input/images/image_000075.JPG'
    im_B_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/BLH0001/input/images/image_000003.JPG'
    # view1, view2, all_kpts1, all_kpts2, all_mconf, mast3r_kpts1, mast3r_kpts2, mast3r_mconf, spider_kpts1, spider_kpts2, spider_mconf = spider_mast3r_match_path(im_A_path, im_B_path, spider_model, mast3r_model, device, coarse_size=512, fine_size=1600)
    # all_kpts1, all_kpts2, all_mconf = all_kpts1.cpu().numpy(), all_kpts2.cpu().numpy(), all_mconf.cpu().numpy()
    # spider_kpts1, spider_kpts2, spider_mconf = spider_kpts1.cpu().numpy(), spider_kpts2.cpu().numpy(), spider_mconf.cpu().numpy()
    # mast3r_kpts1, mast3r_kpts2, mast3r_mconf = mast3r_kpts1.cpu().numpy(), mast3r_kpts2.cpu().numpy(), mast3r_mconf.cpu().numpy()

    # view1, view2, spider_kpts1, spider_kpts2, spider_mconf = spiderfmwarp_match_path(im_A_path, im_B_path, spider_model, device, size=512)
    spider_match_path(im_A_path, im_B_path, spider_model, device, coarse_size=512, fine_size=1600)
    # view1, view2, spider_kpts1, spider_kpts2, spider_mconf = spider_match_path(im_A_path, im_B_path, spider_model, device, coarse_size=512, fine_size=512)
    # spider_kpts1, spider_kpts2, spider_mconf = spider_kpts1.cpu().numpy(), spider_kpts2.cpu().numpy(), spider_mconf.cpu().numpy()
    
    # image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    # image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    # viz_imgs = []
    # for i, view in enumerate([view1, view2]):
    #     rgb_tensor = view['img'][0] * image_std + image_mean
    #     viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    # H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    # img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img = np.concatenate((img0, img1), axis=1)
    # pl.figure()
    # pl.imshow(img)
    # pl.axis('off')  
    # pl.savefig(os.path.join(save_dir, 'raw.png'), dpi=300, bbox_inches='tight')
    # pl.close()
    # def draw_matches(kpts1, kpts2, mconf, name='mast3r_matches.png', n_viz = 400):
    #     valid_matches_im0 = (kpts1[:, 0] >= 3) & (kpts1[:, 0] < int(W0) - 3) & (
    #         kpts1[:, 1] >= 3) & (kpts1[:, 1] < int(H0) - 3)

    #     valid_matches_im1 = (kpts2[:, 0] >= 3) & (kpts2[:, 0] < int(W1) - 3) & (
    #         kpts2[:, 1] >= 3) & (kpts2[:, 1] < int(H1) - 3)
    #     valid_matches_im = mconf > 0.2
    #     valid_matches = valid_matches_im0 & valid_matches_im1 & valid_matches_im
    #     matches_im0, matches_im1 = kpts1[valid_matches], kpts2[valid_matches]
    #     matches_conf = mconf[valid_matches]

    #     # pick n_viz matches
    #     sorted_idx = np.argsort(-matches_conf)
        
    #     topk_idx = sorted_idx[:n_viz]
    #     viz_matches_im0 = matches_im0[topk_idx]
    #     viz_matches_im1 = matches_im1[topk_idx]
    #     viz_conf        = matches_conf[topk_idx]
    #     # pdb.set_trace()
    #     pl.figure()
    #     pl.imshow(img)
    #     cmap = pl.get_cmap('jet')
    #     for i in range(n_viz):
    #         (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    #         pl.plot([x0, x1 + W0], [y0, y1], '-', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False, linewidth=0.3)
            
    #     # pl.show(block=True)
    #     pl.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches='tight')
    #     pl.close()

    # # draw_matches(all_kpts1, all_kpts2, all_mconf, name='matches.png')
    # # draw_matches(mast3r_kpts1, mast3r_kpts2, mast3r_mconf, name='mast3r_matches.png')
    # draw_matches(spider_kpts1, spider_kpts2, spider_mconf, name='spider_matches.png')

    
                
