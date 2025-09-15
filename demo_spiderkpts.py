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
dad_path = os.path.abspath('/cis/home/zshao14/Downloads/dad')
sys.path.insert(0, dad_path)
import dad as dad_detector


detector = dedode_detector_L(weights = None)
detector_dad = dad_detector.load_DaD()
feature_conf = extract_features.confs['superpoint_max']
Model = dynamic_load(extractors, feature_conf["model"]["name"])
sp_model = Model(feature_conf["model"]).eval().to('cpu')



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

def spiderfmwarp_match_path(im_A_path, im_B_path, spider_model, device='cuda', size=512, is_scannet=False):
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
    
    imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=size, intrinsics=None, verbose=False)
    view1, view2 = imgs_coarse
    view1, view2 = collate_with_cat([(view1, view2)])
    
    # spider inference
    corresps12, corresps21 = spiderfmwarp_symmetric_inference(spider_model, view1, view2, device)
    warp0, certainty0 = fmwarp_match(corresps12)
    warp1, certainty1 = fmwarp_match(corresps21, inverse=True)  
    h1, w1 = imgs_coarse[0]['true_shape'][0]
    h2, w2 = imgs_coarse[1]['true_shape'][0]
    spider_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000)
    spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, h1, w1, h2, w2)
    im2_transfer_rgb = F.grid_sample(
                            view2['img'][0][None].to(device), warp0[:,:, 2:][None], mode="bilinear", align_corners=False
                            )[0] ###H1, W1
    im1_transfer_rgb = F.grid_sample(
                            view1['img'][0][None].to(device), warp1[:, :, :2][None], mode="bilinear", align_corners=False
                            )[0] ###H2, W2
    white_im1 = torch.ones((h1, w1)).to(device)
    white_im2 = torch.ones((h2, w2)).to(device)
    # pdb.set_trace()
    vis_im1 = certainty0 * im2_transfer_rgb + (1 - certainty0) * white_im1
    vis_im2 = certainty1 * im1_transfer_rgb + (1 - certainty1) * white_im2
    tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'fmwarp_im1.jpg'))
    tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'fmwarp_im2.jpg'))
    return view1, view2, spider_kpts1, spider_kpts2, spider_mconf

def spider_match_path(save_dir, im_A_path, im_B_path, spider_model, device = 'cuda', coarse_size=512, fine_size=None, name='spider'):
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
        # spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, h1, w1, h2, w2)                                  
 
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
        # spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, h1, w1, h2, w2)
    img1, img2 = cv2.imread(im_A_path), cv2.imread(im_B_path)
    W_A, H_A = Image.open(im_A_path).size
    W_B, H_B = Image.open(im_B_path).size

    spider_matches = spider_matches[spider_mconf >= 0.1]
    spider_mconf = spider_mconf[spider_mconf >= 0.1]
    spider_kpts1, spider_kpts2 = to_pixel_coordinates(spider_matches, H_A, W_A, H_B, W_B)
    canvas = np.zeros((max(H_B, H_A), W_A + W_B, 3), dtype=np.uint8)
    canvas[:H_A, :W_A] = img1
    canvas[:H_B, W_A:] = img2
    offset = np.array([W_A, 0])
    # Draw matches
    num_matches = len(spider_kpts1)
    print(num_matches)

    for i in range(num_matches):
        pt1 = spider_kpts1[i].cpu().numpy().astype('int32')  # Keypoint in img1
        pt2 = spider_kpts2[i].cpu().numpy().astype('int32') + offset  # Keypoint in img2 with offset

        cv2.circle(canvas, tuple(pt1), 2, (0, 0, 255), 1)
        cv2.circle(canvas, tuple(pt2), 2, (0, 0, 255), 1)
        cv2.line(canvas, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

    cv2.imwrite(save_dir +f'/{name}.png', canvas)
    print(f"Finished: {name}")

    detections_A = detector.detect_from_path(im_A_path, num_keypoints = 4096)#10_000)
    detections_B = detector.detect_from_path(im_B_path, num_keypoints = 4096)#10_000)
    dedode_kpts1, dedode_kpts2 = detections_A['keypoints'][0].float(), detections_B['keypoints'][0].float()
    dedode_inds1, dedode_inds2, dedode_matchcerts = match_keypoints(dedode_kpts1, dedode_kpts2, warp0, certainty0, warp1, certainty1)
    dedode_matches0, _ = matches(dedode_inds1, dedode_inds2, dedode_matchcerts, dedode_kpts1, dedode_kpts2)
    d_kpts1 = _to_pixel_coordinates(dedode_kpts1, H_A, W_A).cpu().numpy()
    d_kpts2 = _to_pixel_coordinates(dedode_kpts2, H_B, W_B).cpu().numpy()
    draw(save_dir, d_kpts1, d_kpts2, img1, img2, dedode_matches0, W_A, H_A, W_B, H_B, name=f'dedode_{name}.png')
    print(f"Finished: dedode+{name}")

    detections_dadA = detector_dad.detect_from_path(im_A_path, num_keypoints=4096, return_dense_probs=False,)
    detections_dadB = detector_dad.detect_from_path(im_B_path, num_keypoints=4096, return_dense_probs=False,)
    dad_kpts1, dad_kpts2 = detections_dadA['keypoints'][0].float(), detections_dadB['keypoints'][0].float()
    dad_inds1, dad_inds2, dad_matchcerts = match_keypoints(dad_kpts1, dad_kpts2, warp0, certainty0, warp1, certainty1)
    dad_matches0, _ = matches(dad_inds1, dad_inds2, dad_matchcerts, dad_kpts1, dad_kpts2)
    dad_kpts1 = _to_pixel_coordinates(dad_kpts1, H_A, W_A).cpu().numpy()
    dad_kpts2 = _to_pixel_coordinates(dad_kpts2, H_B, W_B).cpu().numpy()
    draw(save_dir, dad_kpts1, dad_kpts2, img1, img2, dad_matches0, W_A, H_A, W_B, H_B, name=f'dad_{name}.png')
    print(f"Finished: dad+{name}")
   
    pred1 = sp_model({"image": torch.from_numpy(cv2.imread(im_A_path, cv2.IMREAD_GRAYSCALE) / 255.0).float().unsqueeze(0).unsqueeze(0).to('cpu', non_blocking=True)})
    sp_kpts1 = pred1['keypoints'][0]
    pred2 = sp_model({"image": torch.from_numpy(cv2.imread(im_B_path, cv2.IMREAD_GRAYSCALE) / 255.0).float().unsqueeze(0).unsqueeze(0).to('cpu', non_blocking=True)})
    sp_kpts2 = pred2['keypoints'][0]
    sp_kpts1[:,0] /= float(W_A)
    sp_kpts1[:,1] /= float(H_A)
    sp_kpts2[:,0] /= float(W_B)
    sp_kpts2[:,1] /= float(H_B)
    # finally, convert from range (0,1) to (-1,1)
    sp_kpts1 = (sp_kpts1 * 2.) - 1.
    sp_kpts2 = (sp_kpts2 * 2.) - 1.
    sp_kpts1, sp_kpts2 = sp_kpts1.to(device), sp_kpts2.to(device)
    sp_inds1, sp_inds2, sp_matchcerts = match_keypoints(sp_kpts1, sp_kpts2, warp0, certainty0, warp1, certainty1)
    sp_matches0, sp_mscores = matches(sp_inds1, sp_inds2, sp_matchcerts, sp_kpts1, sp_kpts2)
    s_kpts1 = _to_pixel_coordinates(sp_kpts1, H_A, W_A).cpu().numpy()
    s_kpts2 = _to_pixel_coordinates(sp_kpts2, H_B, W_B).cpu().numpy()
    draw(save_dir, s_kpts1, s_kpts2, img1, img2, sp_matches0, W_A, H_A, W_B, H_B, name=f'sp_{name}.png')
    print(f"Finished: sp+{name}")
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
    
    # im2_transfer_rgb = F.grid_sample(
    #                         view2['img'][0][None].to(device), warp0[:,:, 2:][None], mode="bilinear", align_corners=False
    #                         )[0] ###H1, W1
    # im1_transfer_rgb = F.grid_sample(
    #                         view1['img'][0][None].to(device), warp1[:, :, :2][None], mode="bilinear", align_corners=False
    #                         )[0] ###H2, W2
    white_im1 = torch.ones((h1, w1)).to(device)
    white_im2 = torch.ones((h2, w2)).to(device)
    vis_im1 = certainty0 * im2_transfer_rgb + (1 - certainty0) * white_im1
    vis_im2 = certainty1 * im1_transfer_rgb + (1 - certainty1) * white_im2
    tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp_im1.jpg'))
    tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp_im2.jpg'))
    return view1, view2, spider_kpts1, spider_kpts2, spider_mconf

if __name__ == '__main__':
    device = "cuda"
    spider_model = SPIDER.from_pretrained("/cis/home/zshao14/Downloads/spider/model_weights/spider.pth").to(device)
    # pairs = np.load('/cis/net/r24a/data/zshao/data/doppelgangers/pairs_metadata/doppelgangers/pairs_metadata/test_pairs.npy', allow_pickle = True)
    # save_dir = '/cis/home/zshao14/Documents/dg'
    # data_dir = '/cis/net/r24a/data/zshao/data/doppelgangers/test_set/test_set'
    pairs = np.load('/cis/net/r24a/data/zshao/data/doppelgangers/pairs_metadata/doppelgangers/pairs_metadata/train_pairs_megadepth.npy', allow_pickle = True)
    save_dir = '/cis/home/zshao14/Documents/dg/train_megadepth'
    data_dir = '/cis/net/r24a/data/zshao/data/doppelgangers/doppelgangers/images/train_megadepth'
    print(len(pairs))
    for idx, pair in enumerate(pairs):
        if idx % 5000 == 0: 
            im_A_path = os.path.join(data_dir, pair[0])
            im_B_path = os.path.join(data_dir, pair[1])
            name = pair[0].split('/')[0] + str(idx) + '_label' + str(pair[2]) + '_num' + str(pair[3])
            print(name)
            save_path = os.path.join(save_dir, name)
            os.makedirs(save_path, exist_ok=True)
            with torch.no_grad():
                spider_match_path(save_path, im_A_path, im_B_path, spider_model, device, coarse_size=512, fine_size=1600)