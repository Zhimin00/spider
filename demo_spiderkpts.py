import json
from spider.model import SPIDER_two
import torch
import numpy as np

from spider.utils.utils import match, match_upsample, sample_symmetric, to_pixel_coordinates, _to_pixel_coordinates, tensor_to_pil, fmwarp_match, match_keypoints
from spider.utils.image import load_original_images, resize_image_with_intrinsics
import spider.utils.path_to_dust3r #noqa
from dust3r.utils.device import collate_with_cat
from matplotlib import pyplot as pl
import torch.nn.functional as F
import pdb
import sys
import os
import cv2
from PIL import Image
from spider.inference import two_symmetric_inference, two_symmetric_inference_upsample


def matches(inds1, inds2, matchcerts, kpts1, kpts2, threshold=0.1):
    inds1, inds2, matchcerts = map(lambda x: x.detach().cpu().numpy(), (inds1, inds2, matchcerts))
    matches0, mscores = np.full(len(kpts1), -1, dtype=np.int64), np.full(len(kpts1), 0., dtype=np.float32)

    for idx, i1 in enumerate(inds1.tolist()):
        certainty = float(matchcerts[idx])
        if certainty >= threshold:
            matches0[i1] = inds2[idx]
            mscores[i1] = 1.
        else:
            matches0[i1] = -1
            mscores[i1] = 0.
    return matches0, mscores

def draw(output_path, kpts1, kpts2, img1, img2, matches0, W_A, H_A, W_B, H_B, name=''):
    canvas = np.zeros((max(H_B, H_A), W_A + W_B, 3), dtype=np.uint8)
    canvas[:H_A, :W_A] = img1
    canvas[:H_B, W_A:] = img2
    offset = np.array([W_A, 0])
    # Draw keypoints
    for pt in kpts1:
        pt = tuple(pt.astype(int))
        cv2.circle(canvas, pt, radius=2, color=(0, 0, 255), thickness=1)  # Red for kpts1
    for pt in kpts2:
        pt = tuple((pt + offset).astype(int))
        cv2.circle(canvas, pt, radius=2, color=(0, 0, 255), thickness=1)  # Red for kpts2
    # Draw matches
    for i, m in enumerate(matches0):  # matches0 is (1, N)
        if m >= 0:  
            pt1 = tuple(kpts1[i].astype(int))
            pt2 = tuple((kpts2[m] + offset).astype(int))
            cv2.line(canvas, pt1, pt2, color=(0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(output_path, name), canvas)

def visualize_spider_two_match_path(save_dir, im_A_path, im_B_path, model, device = 'cuda', coarse_size=512, fine_size=None, name='spider'):
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
    if fine_size == coarse_size or fine_size is None:
        imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
        view1, view2 = imgs_coarse
        view1, view2 = collate_with_cat([(view1, view2)])
        
        # spider inference
        corresps, res = two_symmetric_inference(model, view1, view2, 'cuda')
        corresps12, corresps21 = corresps
        warp0, certainty0 = match(corresps12)
        warp1, certainty1 = match(corresps21, inverse=True)  
        h1, w1 = imgs_coarse[0]['true_shape'][0]
        h2, w2 = imgs_coarse[1]['true_shape'][0]
        spider_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000)
    else:
        imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
        imgs_fine, _ = resize_image_with_intrinsics(imgs_ori, size=fine_size, intrinsics=None, verbose=False)

        view1_coarse, view2_coarse = imgs_coarse
        view1_coarse, view2_coarse = collate_with_cat([(view1_coarse, view2_coarse)])
        view1, view2 = imgs_fine
        view1, view2 = collate_with_cat([(view1, view2)])
        
        # spider inference
        corresps, res = two_symmetric_inference_upsample(model, view1_coarse, view2_coarse, view1, view2, 'cuda')
        low_corresps12, corresps12, low_corresps21, corresps21 = corresps
        warp0, certainty0 = match_upsample(corresps12, low_corresps12)
        warp1, certainty1 = match_upsample(corresps21, low_corresps21, inverse=True)
        h1, w1 = imgs_fine[0]['true_shape'][0]
        h2, w2 = imgs_fine[1]['true_shape'][0]
        spider_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000, sample_mode = "balanced")
    
    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'][0] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    spider_matches = spider_matches[spider_mconf >= 0.1]
    spider_mconf = spider_mconf[spider_mconf >= 0.1]
    matches_im0, matches_im1 = to_pixel_coordinates(spider_matches, H0, W0, H1, W1)
    matches_im0, matches_im1 = matches_im0.cpu().numpy(), matches_im1.cpu().numpy()
    
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    num_matches = len(matches_im0)
    n_viz = 300
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    # img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img0, img1 = viz_imgs
    gap = 10
    canvas_h = H0 + H1 + gap
    canvas_w = max(W0, W1)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    img0 = (img0 * 255).clip(0, 255).astype(np.uint8)
    img1 = (img1 * 255).clip(0, 255).astype(np.uint8)
    img0= cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    canvas[:H0, :W0] = img0
    offset = np.array([0, H0 + gap])
    canvas[H0+gap:H0+gap+H1, :W1] = img1

    offset = np.array([0, H0 + gap])
    

    for i in range(n_viz):
        pt1 = viz_matches_im0[i].astype('int32')  # Keypoint in img1
        pt2 = viz_matches_im1[i].astype('int32') + offset  # Keypoint in img2 with offset

        cv2.circle(canvas, tuple(pt1), 2, (0, 0, 255), 1)
        cv2.circle(canvas, tuple(pt2), 2, (0, 0, 255), 1)
        cv2.line(canvas, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

    cv2.imwrite(save_dir +f'/{name}.png', canvas)
    print(f"Finished: {name}")

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
    tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp_im1.jpg'))
    tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp_im2.jpg'))
    

if __name__ == '__main__':
    device = "cuda"
    spider_model = SPIDER_two.from_pretrained("./model_weights/spider_two.pth").to(device)
    im_A_path = './assets/M07/M07_B.jpg'
    im_B_path = './assets/M07/M07_B.jpg'
    save_path = './assets/M07'
    os.makedirs(save_path, exist_ok=True)
    visualize_spider_two_match_path(save_path, im_A_path, im_B_path, spider_model, device, coarse_size=512, fine_size=1600)
    