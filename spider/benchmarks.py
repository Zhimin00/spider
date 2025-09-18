import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import cv2
import pdb
from spider.utils.utils import match_keypoints2, compute_relative_pose, estimate_pose, compute_pose_error, pose_auc, match, match_upsample, match_single, match_symmetric, match_symmetric_upsample, sample, sample_symmetric, to_pixel_coordinates, make_symmetric_pairs, tensor_to_pil
from spider.utils.image import load_images_with_intrinsics, load_two_images_with_H, resize_image_with_H, load_images_with_intrinsics_strict, load_two_images_with_H_strict,  load_original_images, resize_image_with_intrinsics
from spider.inference import inference, inference_upsample, crops_inference
from spider.inference import symmetric_inference as spider_symmetric_inference
from spider.inference import symmetric_inference_upsample as spider_symmetric_inference_upsample

from PIL import Image
from mast3r.utils.coarse_to_fine import select_pairs_of_crops, crop_slice
import spider.utils.path_to_dust3r # noqa
from dust3r_visloc.datasets.utils import get_HW_resolution
from dust3r.utils.geometry import geotrf

from mast3r.inference import symmetric_inference as mast3r_symmetric_inference
from mast3r.inference import coarse_to_fine
from mast3r.cloud_opt.sparse_ga import extract_correspondences

import mast3r.utils.path_to_dust3r #noqa
from dust3r.utils.device import collate_with_cat
import sys
dad_path = os.path.abspath('/cis/home/zshao14/Downloads/dad')
sys.path.insert(0, dad_path)
import dad as dad_detector

def crop(img, crop):
    out_cropped_img = img.clone()
    to_orig = torch.eye(3, device=img.device)
    out_cropped_img = img[crop_slice(crop)]
    to_orig[:2, -1] = torch.tensor(crop[:2])
    return out_cropped_img, to_orig

    # Merge all preds
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)
def fine_matching(query_views, map_views, model, device, max_batch_size=48):
    res = crops_inference([query_views, map_views], model, device, batch_size=max_batch_size, verbose=False)
    corresps = res['corresps']
    finest_scale = 1
    im_A_to_im_Bs = corresps[finest_scale]["flow"] 
    certaintys = corresps[finest_scale]["certainty"]
    certainty_s16s= corresps[16]["certainty"]
    # Compute matches
    matches_im_map, matches_im_query, matches_confs = [], [], []
    h0, w0 = query_views['true_shape'][0].cpu().tolist()
    h1, w1 = map_views['true_shape'][0].cpu().tolist() 
    for ppi, (im_A_to_im_B, certainty, certainty_s16) in enumerate(zip(im_A_to_im_Bs, certaintys, certainty_s16s)):
        warp, certainty = match_single(im_A_to_im_B, certainty, certainty_s16)
        sparse_matches, _ = sample(warp, certainty, num=5000)
        kpts_query_ppi, kpts_map_ppi = to_pixel_coordinates(sparse_matches, h0, w0, h1, w1)
        # inverse operation where we uncrop pixel coordinates
        device = map_views['to_orig'][ppi].device
        device = query_views['to_orig'][ppi].device
        matches_im_map_ppi = geotrf(map_views['to_orig'][ppi], kpts_map_ppi, norm=True)
        matches_im_query_ppi = geotrf(query_views['to_orig'][ppi], kpts_query_ppi, norm=True)
        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)

    matches_im_map = torch.cat(matches_im_map, dim=0)
    matches_im_query = torch.cat(matches_im_query, dim=0)
    return matches_im_query, matches_im_map


def dad_spider_match_path(detector, im_A_path, im_B_path, K1_ori, K2_ori, model, device = 'cuda', coarse_size=512, fine_size=None, is_scannet=False):
    dad_kpts0 = detector.detect_from_path(im_A_path, num_keypoints=4096*8, return_dense_probs=False)['keypoints'][0].float()
    dad_kpts1 = detector.detect_from_path(im_B_path, num_keypoints=4096*8, return_dense_probs=False)['keypoints'][0].float()
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
    if fine_size == coarse_size or fine_size is None:
        imgs_coarse, intrinsics = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=[K1_ori, K2_ori], verbose=False)
        K1, K2 = intrinsics
        view1, view2 = imgs_coarse
        view1, view2 = collate_with_cat([(view1, view2)])
        corresps12, corresps21 = spider_symmetric_inference(model, view1, view2, device)
        warp0, certainty0 = match(corresps12)
        warp1, certainty1 = match(corresps21, inverse=True)  
        h1, w1 = imgs_coarse[0]['true_shape'][0]
        h2, w2 = imgs_coarse[1]['true_shape'][0]
    else:
        imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
        imgs_fine, intrinsics = resize_image_with_intrinsics(imgs_ori, size=fine_size, intrinsics=[K1_ori, K2_ori], verbose=False)
        K1, K2 = intrinsics

        view1_coarse, view2_coarse = imgs_coarse
        view1_coarse, view2_coarse = collate_with_cat([(view1_coarse, view2_coarse)])
        view1, view2 = imgs_fine
        view1, view2 = collate_with_cat([(view1, view2)])
        low_corresps12, corresps12, low_corresps21, corresps21 = spider_symmetric_inference_upsample(model, view1_coarse, view2_coarse, view1, view2, device)
        warp0, certainty0 = match_upsample(corresps12, low_corresps12)
        warp1, certainty1 = match_upsample(corresps21, low_corresps21, inverse=True)
        h1, w1 = imgs_fine[0]['true_shape'][0]
        h2, w2 = imgs_fine[1]['true_shape'][0]
    if is_scannet:
        scale1 = 480 / min(w1, h1)
        scale2 = 480 / min(w2, h2)
        w1, h1 = scale1 * w1, scale1 * h1
        w2, h2 = scale2 * w2, scale2 * h2
        K1 = K1 * scale1.item()
        K2 = K2 * scale2.item()
    # sparse_matches, mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000)
    sparse_matches, mconf = match_keypoints2(dad_kpts0, dad_kpts1, warp0, certainty0, warp1, certainty1)
    kpts1, kpts2 = to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
    return kpts1, kpts2, mconf, K1, K2

def spider_match_path(im_A_path, im_B_path, K1_ori, K2_ori, model, device = 'cuda', coarse_size=512, fine_size=None, is_scannet=False):
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
    if fine_size == coarse_size or fine_size is None:
        imgs_coarse, intrinsics = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=[K1_ori, K2_ori], verbose=False)
        K1, K2 = intrinsics
        view1, view2 = imgs_coarse
        view1, view2 = collate_with_cat([(view1, view2)])
        corresps12, corresps21 = spider_symmetric_inference(model, view1, view2, device)
        warp0, certainty0 = match(corresps12)
        warp1, certainty1 = match(corresps21, inverse=True)  
        h1, w1 = imgs_coarse[0]['true_shape'][0]
        h2, w2 = imgs_coarse[1]['true_shape'][0]
    else:
        imgs_coarse, _ = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=None, verbose=False)
        imgs_fine, intrinsics = resize_image_with_intrinsics(imgs_ori, size=fine_size, intrinsics=[K1_ori, K2_ori], verbose=False)
        K1, K2 = intrinsics

        view1_coarse, view2_coarse = imgs_coarse
        view1_coarse, view2_coarse = collate_with_cat([(view1_coarse, view2_coarse)])
        view1, view2 = imgs_fine
        view1, view2 = collate_with_cat([(view1, view2)])
        low_corresps12, corresps12, low_corresps21, corresps21 = spider_symmetric_inference_upsample(model, view1_coarse, view2_coarse, view1, view2, device)
        warp0, certainty0 = match_upsample(corresps12, low_corresps12)
        warp1, certainty1 = match_upsample(corresps21, low_corresps21, inverse=True)
        h1, w1 = imgs_fine[0]['true_shape'][0]
        h2, w2 = imgs_fine[1]['true_shape'][0]
    if is_scannet:
        scale1 = 480 / min(w1, h1)
        scale2 = 480 / min(w2, h2)
        w1, h1 = scale1 * w1, scale1 * h1
        w2, h2 = scale2 * w2, scale2 * h2
        K1 = K1 * scale1.item()
        K2 = K2 * scale2.item()
    sparse_matches, mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=5000)
    kpts1, kpts2 = to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
    return kpts1, kpts2, mconf, K1, K2

class AerialMegaDepthPoseEstimationBenchmark:
    def __init__(self, data_root="data/megadepth") -> None:
        self.data_root = data_root
        with np.load(os.path.join(self.data_root, 'aerial_megadepth_test_scenes0015_0022.npz'), allow_pickle=True) as data:
            self.all_scenes = data['scenes']
            self.all_images = data['images']
            self.pairs = data['pairs']

    def load_intrinsics_and_pose(self, npz_path):
        camera_params = np.load(npz_path)
        K = camera_params["intrinsics"].astype(np.float32)
        T = camera_params["cam2world"].astype(np.float32)
        T_inv = np.linalg.inv(T)  
        return K, T_inv

    def benchmark(self, model, device='cuda', model_name = 'spider', debug=False, coarse_size=512, fine_size=1344):
        detector = dad_detector.load_DaD()
        with torch.no_grad():
            data_root = self.data_root
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            thresholds = [5, 10, 20]

            pair_inds = range(len(self.pairs))
            for pairind in tqdm(pair_inds):
                scene_id, idx1, idx2, score = self.pairs[pairind]

                scene = self.all_scenes[scene_id]
                seq_path = f"{data_root}/{scene}"
                im_A_name, im_B_name = self.all_images[idx1], self.all_images[idx2]
                
                ## load camera parameters
                K1_ori, T1 = self.load_intrinsics_and_pose(os.path.join(seq_path, im_A_name + ".npz"))
                R1, t1 = T1[:3, :3], T1[:3, 3]
                K2_ori, T2 = self.load_intrinsics_and_pose(os.path.join(seq_path, im_B_name + ".npz"))
                R2, t2 = T2[:3, :3], T2[:3, 3]
                R, t = compute_relative_pose(R1, t1, R2, t2)
                T1_to_2 = np.concatenate((R,t[:,None]), axis=-1)
                
                im_A_path = f"{seq_path}/{im_A_name}.jpg"
                im_B_path = f"{seq_path}/{im_B_name}.jpg"

                # kpts1, kpts2, mconf, K1, K2 = spider_match_path(im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size)
                kpts1, kpts2, mconf, K1, K2 = dad_spider_match_path(detector, im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size)
                
                kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                for _ in range(5):
                    shuffling = np.random.permutation(np.arange(len(kpts1)))
                    kpts1 = kpts1[shuffling]
                    kpts2 = kpts2[shuffling]
                    try:
                        threshold = 0.5 
                        norm_threshold = threshold / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                        R_est, t_est, mask = estimate_pose(
                            kpts1,
                            kpts2,
                            K1,
                            K2,
                            norm_threshold,
                            conf=0.99999,
                        )
                        T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
                        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                        e_pose = max(e_t, e_R)
                    except Exception as e:
                        print(repr(e))
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)
                # pdb.set_trace()
            tot_e_pose = np.array(tot_e_pose)
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            print(f"{model_name} aerial auc: {auc}")
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }

class MegaDepthPoseEstimationBenchmark:
    def __init__(self, data_root="data/megadepth", scene_names = None) -> None:
        if scene_names is None:
            self.scene_names = [
                "0015_0.1_0.3.npz",
                "0015_0.3_0.5.npz",
                "0022_0.1_0.3.npz",
                "0022_0.3_0.5.npz",
                "0022_0.5_0.7.npz",
            ]
        else:
            self.scene_names = scene_names
        self.scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]
        self.data_root = data_root

    def benchmark(self, model, device='cuda', model_name = 'spider', debug=False, coarse_size=512, fine_size=1344):
        detector = dad_detector.load_DaD()
        with torch.no_grad():
            data_root = self.data_root
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            thresholds = [5, 10, 20]
            for scene_ind in range(len(self.scenes)):
                import os
                scene_name = os.path.splitext(self.scene_names[scene_ind])[0]
                scene = self.scenes[scene_ind]
                pairs = scene["pair_infos"]
                intrinsics = scene["intrinsics"]
                poses = scene["poses"]
                im_paths = scene["image_paths"]
                pair_inds = range(len(pairs))
                for pairind in tqdm(pair_inds):
                    idx1, idx2 = pairs[pairind][0]
                    K1_ori = intrinsics[idx1].copy()
                    T1 = poses[idx1].copy()
                    R1, t1 = T1[:3, :3], T1[:3, 3]
                    K2_ori = intrinsics[idx2].copy()
                    T2 = poses[idx2].copy()
                    R2, t2 = T2[:3, :3], T2[:3, 3]
                    R, t = compute_relative_pose(R1, t1, R2, t2)
                    T1_to_2 = np.concatenate((R,t[:,None]), axis=-1)
                    im_A_path = f"{data_root}/{im_paths[idx1]}"
                    im_B_path = f"{data_root}/{im_paths[idx2]}"
                    # kpts1, kpts2, mconf, K1, K2 = spider_match_path(im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size)
                    kpts1, kpts2, mconf, K1, K2 = dad_spider_match_path(detector, im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size)
                
                    kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                    # imgs, _ = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=coarse_size, intrinsics=None)
                    # imgs_large, new_intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=fine_size, intrinsics=[K1_ori, K2_ori])
                    # image_pairs = make_symmetric_pairs(imgs)
                    # image_large_pairs = make_symmetric_pairs(imgs_large)
                    # res = inference_upsample(image_pairs, image_large_pairs, model, device, batch_size=1, verbose=True)
                    # warp1, certainty1, warp2, certainty2 = match_symmetric_upsample(res['corresps'], res['low_corresps'])
                    # sparse_matches, _ = sample_symmetric(warp1, certainty1, warp2, certainty2, num=5000)
                    # K1, K2 = new_intrinsics
                    # h1, w1 = imgs_large[0]['true_shape'][0]
                    # h2, w2 = imgs_large[1]['true_shape'][0]
                    if debug:
                        from matplotlib import pyplot as pl
                        save_dir = '/cis/home/zshao14/Downloads/spider/assets/mega_benchmark'
                        os.makedirs(save_dir, exist_ok=True)
                        view1, view2 = res['view1'], res['view2']
                        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
                        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

                        viz_imgs = []
                        for i, view in enumerate([view1, view2]):
                            rgb_tensor = view['img'][0] * image_std + image_mean
                            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

                        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                        kpts1, kpts2 = to_pixel_coordinates(sparse_matches, H0, W0, H1, W1)
                        matches_im0, matches_im1 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                        
                        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

                        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

                        valid_matches = valid_matches_im0 & valid_matches_im1
                        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

                        num_matches = len(matches_im0)
                        n_viz = 20
                        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
                        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                        img = np.concatenate((img0, img1), axis=1)
                        pl.figure()
                        pl.imshow(img)
                        pl.axis('off')  
                        pl.savefig(os.path.join(save_dir, 'raw.png'), dpi=300, bbox_inches='tight')
                        pl.close()


                        pl.figure()
                        pl.imshow(img)
                        cmap = pl.get_cmap('jet')
                        for i in range(n_viz):
                            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
                            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                        # pl.show(block=True)
                        pl.savefig(os.path.join(save_dir, 'matches.png'), dpi=300, bbox_inches='tight')
                        pl.close()
                        im2_transfer_rgb = F.grid_sample(
                            view2['img'][0][None], warp1[:,:, 2:][None], mode="bilinear", align_corners=False
                            )[0] ###H1, W1
                        im1_transfer_rgb = F.grid_sample(
                            view1['img'][0][None], warp2[:, :, :2][None], mode="bilinear", align_corners=False
                            )[0] ###H2, W2
                        white_im1 = torch.ones((H0,W0))
                        white_im2 = torch.ones((H1,W1))
                        vis_im1 = certainty1 * im2_transfer_rgb + (1 - certainty1) * white_im1
                        vis_im2 = certainty2 * im1_transfer_rgb + (1 - certainty2) * white_im2
                        tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp_im1.jpg'))
                        tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp_im2.jpg'))

                    # if True: # Note: we keep this true as it was used in DKM/RoMa papers. There is very little difference compared to setting to False. 
                    #     scale1 = 1200 / max(w1, h1)
                    #     scale2 = 1200 / max(w2, h2)
                    #     w1, h1 = scale1 * w1, scale1 * h1
                    #     w2, h2 = scale2 * w2, scale2 * h2
                    #     K1, K2 = K1.copy(), K2.copy()
                    #     K1[:2] = K1[:2] * scale1
                    #     K2[:2] = K2[:2] * scale2
                    for _ in range(5):
                        shuffling = np.random.permutation(np.arange(len(kpts1)))
                        kpts1 = kpts1[shuffling]
                        kpts2 = kpts2[shuffling]
                        try:
                            threshold = 0.5 
                            norm_threshold = threshold / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                            R_est, t_est, mask = estimate_pose(
                                kpts1,
                                kpts2,
                                K1,
                                K2,
                                norm_threshold,
                                conf=0.99999,
                            )
                            T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
                            e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                            e_pose = max(e_t, e_R)
                        except Exception as e:
                            print(repr(e))
                            e_t, e_R = 90, 90
                            e_pose = max(e_t, e_R)
                        tot_e_t.append(e_t)
                        tot_e_R.append(e_R)
                        tot_e_pose.append(e_pose)
                    # pdb.set_trace()
            tot_e_pose = np.array(tot_e_pose)
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            print(f"{model_name} megadepth auc: {auc}")
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    def benchmark_2dand3d(self, model1, model2, device='cuda', model_name = 'spider', debug=False, coarse_size=512, fine_size=1344):
        model1.eval()
        model2.eval()
        with torch.no_grad():
            data_root = self.data_root
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            thresholds = [5, 10, 20]
            for scene_ind in range(len(self.scenes)):
                import os
                scene_name = os.path.splitext(self.scene_names[scene_ind])[0]
                scene = self.scenes[scene_ind]
                pairs = scene["pair_infos"]
                intrinsics = scene["intrinsics"]
                poses = scene["poses"]
                im_paths = scene["image_paths"]
                pair_inds = range(len(pairs))
                for pairind in tqdm(pair_inds):
                    idx1, idx2 = pairs[pairind][0]
                    K1_ori = intrinsics[idx1].copy()
                    T1 = poses[idx1].copy()
                    R1, t1 = T1[:3, :3], T1[:3, 3]
                    K2_ori = intrinsics[idx2].copy()
                    T2 = poses[idx2].copy()
                    R2, t2 = T2[:3, :3], T2[:3, 3]
                    R, t = compute_relative_pose(R1, t1, R2, t2)
                    T1_to_2 = np.concatenate((R,t[:,None]), axis=-1)
                    im_A_path = f"{data_root}/{im_paths[idx1]}"
                    im_B_path = f"{data_root}/{im_paths[idx2]}"
                    
                    imgs, new_intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=coarse_size, intrinsics=[K1_ori, K2_ori])
                    # K1, K2 = new_intrinsics
                    # # if np.all(imgs[0]['true_shape'] == imgs[1]['true_shape']):
                    # #     continue
                    image_pairs = make_symmetric_pairs(imgs)
                    res = inference(image_pairs, model1, device, batch_size=1, verbose=True)
                    warp1, certainty1, warp2, certainty2 = match_symmetric(res['corresps'])
                    sparse_matches, _ = sample_symmetric(warp1, certainty1, warp2, certainty2, num=5000)
                    view1, view2 = imgs
                    view1, view2 = collate_with_cat([(view1, view2)])
                    res = mast3r_symmetric_inference(model2, view1, view2, device)
                    descs = [r['desc'][0] for r in res]
                    qonfs = [r['desc_conf'][0] for r in res]  
                    # perform reciprocal matching
                    corres = extract_correspondences(descs, qonfs, device=device, subsample=8)
                    
                    mast3r_kpts1, mast3r_kpts2, mast3r_mconf = corres                                      
                    mast3r_kpts1 = mast3r_kpts1 + 0.5
                    mast3r_kpts2 = mast3r_kpts2 + 0.5

                    imgs_large, new_intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=fine_size, intrinsics=[K1_ori, K2_ori])
                    K1, K2 = new_intrinsics
                    h1_coarse, w1_coarse = imgs[0]['true_shape'][0]
                    h2_coarse, w2_coarse = imgs[1]['true_shape'][0]

                    h1, w1 = imgs_large[0]['true_shape'][0]
                    h2, w2 = imgs_large[1]['true_shape'][0]
                    mast3r_kpts1 = (
                        torch.stack(
                            (
                                (w1 / w1_coarse) * (mast3r_kpts1[..., 0]),
                                (h1 / h1_coarse) * (mast3r_kpts1[..., 1]),
                            ),
                            axis=-1,
                        )
                    )
                    mast3r_kpts2 = (
                        torch.stack(
                            (
                                (w2 / w2_coarse) * (mast3r_kpts2[..., 0]),
                                (h2 / h2_coarse) * (mast3r_kpts2[..., 1]),
                            ),
                            axis=-1,
                        )
                    )

                    
                    # image_pairs = make_symmetric_pairs(imgs)
                    image_large_pairs = make_symmetric_pairs(imgs_large)
                    res = inference_upsample(image_pairs, image_large_pairs, model1, device, batch_size=1, verbose=True)
                    warp1, certainty1, warp2, certainty2 = match_symmetric_upsample(res['corresps'], res['low_corresps'])
                    sparse_matches, _ = sample_symmetric(warp1, certainty1, warp2, certainty2, num=5000)
                    
                    # h1, w1 = imgs_large[0]['true_shape'][0]
                    # h2, w2 = imgs_large[1]['true_shape'][0]
                    if debug:
                        from matplotlib import pyplot as pl
                        save_dir = '/cis/home/zshao14/Downloads/spider/assets/mega_benchmark'
                        os.makedirs(save_dir, exist_ok=True)
                        view1, view2 = res['view1'], res['view2']
                        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
                        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

                        viz_imgs = []
                        for i, view in enumerate([view1, view2]):
                            rgb_tensor = view['img'][0] * image_std + image_mean
                            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

                        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                        kpts1, kpts2 = to_pixel_coordinates(sparse_matches, H0, W0, H1, W1)
                        matches_im0, matches_im1 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                        
                        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

                        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

                        valid_matches = valid_matches_im0 & valid_matches_im1
                        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

                        num_matches = len(matches_im0)
                        n_viz = 20
                        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
                        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                        img = np.concatenate((img0, img1), axis=1)
                        pl.figure()
                        pl.imshow(img)
                        pl.axis('off')  
                        pl.savefig(os.path.join(save_dir, 'raw.png'), dpi=300, bbox_inches='tight')
                        pl.close()


                        pl.figure()
                        pl.imshow(img)
                        cmap = pl.get_cmap('jet')
                        for i in range(n_viz):
                            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
                            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                        # pl.show(block=True)
                        pl.savefig(os.path.join(save_dir, 'matches.png'), dpi=300, bbox_inches='tight')
                        pl.close()
                        im2_transfer_rgb = F.grid_sample(
                            view2['img'][0][None], warp1[:,:, 2:][None], mode="bilinear", align_corners=False
                            )[0] ###H1, W1
                        im1_transfer_rgb = F.grid_sample(
                            view1['img'][0][None], warp2[:, :, :2][None], mode="bilinear", align_corners=False
                            )[0] ###H2, W2
                        white_im1 = torch.ones((H0,W0))
                        white_im2 = torch.ones((H1,W1))
                        vis_im1 = certainty1 * im2_transfer_rgb + (1 - certainty1) * white_im1
                        vis_im2 = certainty2 * im1_transfer_rgb + (1 - certainty2) * white_im2
                        tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp_im1.jpg'))
                        tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp_im2.jpg'))

                    # if True: # Note: we keep this true as it was used in DKM/RoMa papers. There is very little difference compared to setting to False. 
                    #     scale1 = 1200 / max(w1, h1)
                    #     scale2 = 1200 / max(w2, h2)
                    #     w1, h1 = scale1 * w1, scale1 * h1
                    #     w2, h2 = scale2 * w2, scale2 * h2
                    #     K1, K2 = K1.copy(), K2.copy()
                    #     K1[:2] = K1[:2] * scale1
                    #     K2[:2] = K2[:2] * scale2
                    # h1, w1 = imgs[0]['true_shape'][0]
                    # h2, w2 = imgs[1]['true_shape'][0]
                    spider_kpts1, spider_kpts2 = to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
                    mast3r_kpts1, mast3r_kpts2 = mast3r_kpts1.cpu(), mast3r_kpts2.cpu()
                    kpts1 = torch.cat((spider_kpts1, mast3r_kpts1), dim=0)
                    kpts2 = torch.cat((spider_kpts2, mast3r_kpts2), dim=0)
                    
                    kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                    for _ in range(5):
                        shuffling = np.random.permutation(np.arange(len(kpts1)))
                        kpts1 = kpts1[shuffling]
                        kpts2 = kpts2[shuffling]
                        try:
                            threshold = 0.5 
                            norm_threshold = threshold / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                            R_est, t_est, mask = estimate_pose(
                                kpts1,
                                kpts2,
                                K1,
                                K2,
                                norm_threshold,
                                conf=0.99999,
                            )
                            T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
                            e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                            e_pose = max(e_t, e_R)
                        except Exception as e:
                            print(repr(e))
                            e_t, e_R = 90, 90
                            e_pose = max(e_t, e_R)
                        tot_e_t.append(e_t)
                        tot_e_R.append(e_R)
                        tot_e_pose.append(e_pose)
                    # pdb.set_trace()
            tot_e_pose = np.array(tot_e_pose)
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            print(f"{model_name} auc: {auc}")
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    def benchmark_coarse_to_fine(self, model, device='cuda', model_name = 'spider', debug=False):
        model.eval()
        with torch.no_grad():
            data_root = self.data_root
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            thresholds = [5, 10, 20]
            for scene_ind in range(len(self.scenes)):
                import os
                scene_name = os.path.splitext(self.scene_names[scene_ind])[0]
                scene = self.scenes[scene_ind]
                pairs = scene["pair_infos"]
                intrinsics = scene["intrinsics"]
                poses = scene["poses"]
                im_paths = scene["image_paths"]
                pair_inds = range(len(pairs))
                for pairind in tqdm(pair_inds):
                    idx1, idx2 = pairs[pairind][0]
                    K1_ori = intrinsics[idx1].copy()
                    T1 = poses[idx1].copy()
                    R1, t1 = T1[:3, :3], T1[:3, 3]
                    K2_ori = intrinsics[idx2].copy()
                    T2 = poses[idx2].copy()
                    R2, t2 = T2[:3, :3], T2[:3, 3]
                    R, t = compute_relative_pose(R1, t1, R2, t2)
                    T1_to_2 = np.concatenate((R,t[:,None]), axis=-1)
                    im_A_path = f"{data_root}/{im_paths[idx1]}"
                    im_B_path = f"{data_root}/{im_paths[idx2]}"
                    
                    imgs, _ = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=512, intrinsics=None)
                    image_pairs = make_symmetric_pairs(imgs)

                    # coarse_res = inference(image_pairs,model, device, batch_size=1, verbose=True)
                    # warp1, certainty1, warp2, certainty2 = match_symmetric(coarse_res['corresps'])
                    # sparse_matches, _ = sample_symmetric(warp1, certainty1, warp2, certainty2, num=5000)
                    imgs_large, new_intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=1344, intrinsics=[K1_ori, K2_ori])
                    
                    image_large_pairs = make_symmetric_pairs(imgs_large)
                    res = inference_upsample(image_pairs, image_large_pairs, model, device, batch_size=1, verbose=True)
                    warp1, certainty1, warp2, certainty2 = match_symmetric_upsample(res['corresps'], res['low_corresps'])
                    sparse_matches, _ = sample_symmetric(warp1, certainty1, warp2, certainty2, num=5000)
                    
                    h1, w1 = imgs_large[0]['true_shape'][0]
                    h2, w2 = imgs_large[1]['true_shape'][0]
                    kpts1, kpts2 = to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
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
                        continue
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
                        kpts1, kpts2 = fine_matching(query_crop_view, map_crop_view, model, device)                                                  
                    
                    kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                    K1, K2 = new_intrinsics
                    if debug:
                        from matplotlib import pyplot as pl
                        save_dir = '/cis/home/zshao14/Downloads/spider/assets/mega_benchmark'
                        os.makedirs(save_dir, exist_ok=True)
                        view1, view2 = imgs_large
                        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
                        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

                        viz_imgs = []
                        for i, view in enumerate([view1, view2]):
                            rgb_tensor = view['img'][0] * image_std + image_mean
                            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

                        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                        pdb.set_trace()
                        matches_im0, matches_im1 = kpts1, kpts2
                        
                        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

                        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

                        valid_matches = valid_matches_im0 & valid_matches_im1
                        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

                        num_matches = len(matches_im0)
                        n_viz = 20
                        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
                        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                        img = np.concatenate((img0, img1), axis=1)
                        pl.figure()
                        pl.imshow(img)
                        pl.axis('off')  
                        pl.savefig(os.path.join(save_dir, 'raw.png'), dpi=300, bbox_inches='tight')
                        pl.close()


                        pl.figure()
                        pl.imshow(img)
                        cmap = pl.get_cmap('jet')
                        for i in range(n_viz):
                            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
                            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                        # pl.show(block=True)
                        pl.savefig(os.path.join(save_dir, 'matches.png'), dpi=300, bbox_inches='tight')
                        pl.close()
                        # im2_transfer_rgb = F.grid_sample(
                        #     view2['img'][0][None], warp1[:,:, 2:][None], mode="bilinear", align_corners=False
                        #     )[0] ###H1, W1
                        # im1_transfer_rgb = F.grid_sample(
                        #     view1['img'][0][None], warp2[:, :, :2][None], mode="bilinear", align_corners=False
                        #     )[0] ###H2, W2
                        # white_im1 = torch.ones((H0,W0))
                        # white_im2 = torch.ones((H1,W1))
                        # vis_im1 = certainty1 * im2_transfer_rgb + (1 - certainty1) * white_im1
                        # vis_im2 = certainty2 * im1_transfer_rgb + (1 - certainty2) * white_im2
                        # tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp_im1.jpg'))
                        # tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp_im2.jpg'))

                    # if True: # Note: we keep this true as it was used in DKM/RoMa papers. There is very little difference compared to setting to False. 
                    #     scale1 = 1200 / max(w1, h1)
                    #     scale2 = 1200 / max(w2, h2)
                    #     w1, h1 = scale1 * w1, scale1 * h1
                    #     w2, h2 = scale2 * w2, scale2 * h2
                    #     K1, K2 = K1.copy(), K2.copy()
                    #     K1[:2] = K1[:2] * scale1
                    #     K2[:2] = K2[:2] * scale2

                    
                    for _ in range(5):
                        shuffling = np.random.permutation(np.arange(len(kpts1)))
                        kpts1 = kpts1[shuffling]
                        kpts2 = kpts2[shuffling]
                        try:
                            threshold = 0.5 
                            norm_threshold = threshold / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                            R_est, t_est, mask = estimate_pose(
                                kpts1,
                                kpts2,
                                K1,
                                K2,
                                norm_threshold,
                                conf=0.99999,
                            )
                            T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
                            e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                            e_pose = max(e_t, e_R)
                        except Exception as e:
                            print(repr(e))
                            e_t, e_R = 90, 90
                            e_pose = max(e_t, e_R)
                        tot_e_t.append(e_t)
                        tot_e_R.append(e_R)
                        tot_e_pose.append(e_pose)
            tot_e_pose = np.array(tot_e_pose)
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            print(f"{model_name} auc: {auc}")
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
        
        
class ScanNetBenchmark:
    def __init__(self, data_root="data/scannet") -> None:
        self.data_root = data_root

    def benchmark(self, model, device='cuda', model_name = 'spider', debug=False, coarse_size=512, fine_size=1344):
        detector = dad_detector.load_DaD()
        with torch.no_grad():
            data_root = self.data_root
            tmp = np.load(osp.join(data_root, "test.npz"))
            pairs, rel_pose = tmp["name"], tmp["rel_pose"]
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            pair_inds = np.random.choice(
                range(len(pairs)), size=len(pairs), replace=False
            )
            for pairind in tqdm(pair_inds, smoothing=0.9):
                scene = pairs[pairind]
                scene_name = f"scene0{scene[0]}_00"
                im_A_path = osp.join(
                        self.data_root,
                        "scannet_test_1500",
                        scene_name,
                        "color",
                        f"{scene[2]}.jpg",
                    )
                im_B_path = osp.join(
                        self.data_root,
                        "scannet_test_1500",
                        scene_name,
                        "color",
                        f"{scene[3]}.jpg",
                    )
                T_gt = rel_pose[pairind].reshape(3, 4)
                R, t = T_gt[:3, :3], T_gt[:3, 3]
                K = np.stack(
                    [
                        np.array([float(i) for i in r.split()])
                        for r in open(
                            osp.join(
                                self.data_root,
                                "scannet_test_1500",
                                scene_name,
                                "intrinsic",
                                "intrinsic_color.txt",
                            ),
                            "r",
                        )
                        .read()
                        .split("\n")
                        if r
                    ]
                )

                K1_ori = K.copy()
                K2_ori = K.copy()
                # if fine_size == coarse_size:
                #     imgs, intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=coarse_size, intrinsics=[K1_ori, K2_ori])
                #     image_pairs = make_symmetric_pairs(imgs)
                #     res = inference(image_pairs, model, device, batch_size=1, verbose=True)
                #     warp1, certainty1, warp2, certainty2 = match_symmetric(res['corresps'])
                #     sparse_matches, _ = sample_symmetric(warp1, certainty1, warp2, certainty2, num=5000)
                #     K1, K2 = intrinsics
                #     h1, w1 = imgs[0]['true_shape'][0]
                #     h2, w2 = imgs[1]['true_shape'][0]

                # else:
                # imgs, _ = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=coarse_size, intrinsics=None)
                # imgs_large, intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=fine_size, intrinsics=[K1_ori, K2_ori])
                # # if np.all(imgs[0]['true_shape'] == imgs[1]['true_shape']):
                # #     continue
                # image_pairs = make_symmetric_pairs(imgs)
                # image_large_pairs = make_symmetric_pairs(imgs_large)
                # res = inference_upsample(image_pairs, image_large_pairs, model, device, batch_size=1, verbose=True)
                # warp1, certainty1, warp2, certainty2 = match_symmetric_upsample(res['corresps'], res['low_corresps'])
            
                # sparse_matches, _ = sample_symmetric(warp1, certainty1, warp2, certainty2, num=5000)
                # K1, K2 = intrinsics
                # h1, w1 = imgs_large[0]['true_shape'][0]
                # h2, w2 = imgs_large[1]['true_shape'][0]
                if debug:
                    from matplotlib import pyplot as pl
                    save_dir = '/cis/home/zshao14/Downloads/spider/assets/scannet_benchmark'
                    os.makedirs(save_dir, exist_ok=True)
                    view1, view2 = res['view1'], res['view2']
                    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
                    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

                    viz_imgs = []
                    for i, view in enumerate([view1, view2]):
                        rgb_tensor = view['img'][0] * image_std + image_mean
                        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

                    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                    kpts1, kpts2 = to_pixel_coordinates(sparse_matches, H0, W0, H1, W1)
                    matches_im0, matches_im1 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                    
                    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

                    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

                    valid_matches = valid_matches_im0 & valid_matches_im1
                    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

                    num_matches = len(matches_im0)
                    n_viz = 20
                    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
                    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                    img = np.concatenate((img0, img1), axis=1)
                    pl.figure()
                    pl.imshow(img)
                    pl.axis('off')  
                    pl.savefig(os.path.join(save_dir, 'raw.png'), dpi=300, bbox_inches='tight')
                    pl.close()

                    pl.figure()
                    pl.imshow(img)
                    cmap = pl.get_cmap('jet')
                    for i in range(n_viz):
                        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
                        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                    # pl.show(block=True)
                    pl.savefig(os.path.join(save_dir, 'matches.png'), dpi=300, bbox_inches='tight')
                    pl.close()
                    im2_transfer_rgb = F.grid_sample(
                        view2['img'][0][None], warp1[:,:, 2:][None], mode="bilinear", align_corners=False
                        )[0] ###H1, W1
                    im1_transfer_rgb = F.grid_sample(
                        view1['img'][0][None], warp2[:, :, :2][None], mode="bilinear", align_corners=False
                        )[0] ###H2, W2
                    white_im1 = torch.ones((H0,W0))
                    white_im2 = torch.ones((H1,W1))
                    vis_im1 = certainty1 * im2_transfer_rgb + (1 - certainty1) * white_im1
                    vis_im2 = certainty2 * im1_transfer_rgb + (1 - certainty2) * white_im2
                    tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp_im1.jpg'))
                    tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp_im2.jpg'))


                # kpts1, kpts2 = to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
                              

                offset = 0.5
                # kpts1, kpts2, mconf, K1, K2 = spider_match_path(im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size, is_scannet=True)
                kpts1, kpts2, mconf, K1, K2 = dad_spider_match_path(detector, im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size)
                
                kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                # kpts1 = sparse_matches[:, :2]
                # kpts1 = kpts1.cpu().numpy()
                # kpts1 = (
                #     np.stack(
                #         (
                #             w1 * (kpts1[:, 0] + 1) / 2 - offset,
                #             h1 * (kpts1[:, 1] + 1) / 2 - offset,
                #         ),
                #         axis=-1,
                #     )
                # )
                # kpts2 = sparse_matches[:, 2:]
                # kpts2 = kpts2.cpu().numpy()
                # kpts2 = (
                #     np.stack(
                #         (
                #             w2 * (kpts2[:, 0] + 1) / 2 - offset,
                #             h2 * (kpts2[:, 1] + 1) / 2 - offset,
                #         ),
                #         axis=-1,
                #     )
                # )
                for _ in range(5):
                    shuffling = np.random.permutation(np.arange(len(kpts1)))
                    kpts1 = kpts1[shuffling]
                    kpts2 = kpts2[shuffling]
                    try:
                        norm_threshold = 0.5 / (
                        np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                        R_est, t_est, mask = estimate_pose(
                            kpts1,
                            kpts2,
                            K1,
                            K2,
                            norm_threshold,
                            conf=0.99999,
                        )
                        T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  #
                        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                        e_pose = max(e_t, e_R)
                    except Exception as e:
                        print(repr(e))
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)
                # pdb.set_trace()
            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            print(f"{model_name} scannet auc: {auc}")
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
        
class HpatchesHomogBenchmark:
    """Hpatches grid goes from [0,n-1] instead of [0.5,n-0.5]"""

    def __init__(self, dataset_path, seqs_dir = "hpatches-sequences-release") -> None:
        # seqs_dir = "hpatches-sequences-release"
        self.seqs_path = os.path.join(dataset_path, seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))
        # Ignore seqs is same as LoFTR.
        self.ignore_seqs = set(
            [
                "i_contruction",
                "i_crownnight",
                "i_dc",
                "i_pencils",
                "i_whitebuilding",
                "v_artisans",
                "v_astronautis",
                "v_talent",
            ]
        )

    def convert_coordinates(self, im_A_coords, im_A_to_im_B, wq, hq, wsup, hsup):
        offset = 0.5  # Hpatches assumes that the center of the top-left pixel is at [0,0] (I think)
        im_A_coords = (
            np.stack(
                (
                    wq * (im_A_coords[..., 0] + 1) / 2,
                    hq * (im_A_coords[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        im_A_to_im_B = (
            np.stack(
                (
                    wsup * (im_A_to_im_B[..., 0] + 1) / 2,
                    hsup * (im_A_to_im_B[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        return im_A_coords, im_A_to_im_B

    def benchmark(self, model, device='cuda', model_name = 'spider', debug=False, coarse_size=512, fine_size=1344):
        homog_dists = []
        
        for seq_idx, seq_name in tqdm(
            enumerate(self.seq_names), total=len(self.seq_names)
        ):
            im_A_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im_A = Image.open(im_A_path)
            w1, h1 = im_A.size
            for im_idx in range(2, 7):
                im_B_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                im_B = Image.open(im_B_path)
                w2, h2 = im_B.size
                H = np.loadtxt(
                    os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx))
                )
                # pos_a, pos_b, mconf, H_new, h1, w1, h2, w2 = spider_match_path_H(im_A_path, im_B_path, H, model, device, coarse_size=coarse_size, fine_size=fine_size)
                # pos_a, pos_b, mconf = pos_a.cpu().numpy(), pos_b.cpu().numpy(), mconf.cpu().numpy()

                sparse_matches, mconf = spider_match_path_H(im_A_path, im_B_path, H, model, device, coarse_size=coarse_size, fine_size=fine_size)
                sparse_matches = sparse_matches.cpu().numpy()
                pos_a, pos_b = self.convert_coordinates(sparse_matches[:, :2], sparse_matches[:, 2:], w1, h1, w2, h2)
                try:
                    H_pred, inliers = cv2.findHomography(
                        pos_a,
                        pos_b,
                        method = cv2.RANSAC,
                        confidence = 0.99999,
                        ransacReprojThreshold = 3 * min(w2, h2) / 480,
                    )
                except:
                    H_pred = None
                if H_pred is None:
                    H_pred = np.zeros((3, 3))
                    H_pred[2, 2] = 1.0
                corners = np.array(
                    [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]]
                )
                real_warped_corners = np.dot(corners, np.transpose(H))
                real_warped_corners = (
                    real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                )
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                mean_dist = np.mean(
                    np.linalg.norm(real_warped_corners - warped_corners, axis=1)
                ) / (min(w2, h2) / 480.0)
                homog_dists.append(mean_dist)
                # pdb.set_trace()

        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        auc = pose_auc(np.array(homog_dists), thresholds)
        print(auc)
        homog_dists = np.array(homog_dists)
        acc_3 = (homog_dists < 3).mean()
        acc_5 = (homog_dists < 5).mean()
        acc_10 = (homog_dists < 10).mean()
        print(acc_3, acc_5, acc_10)
        pdb.set_trace()
        return {
            "hpatches_homog_auc_3": auc[2],
            "hpatches_homog_auc_5": auc[4],
            "hpatches_homog_auc_10": auc[9],
        }
        
def spider_match_path_H(im_A_path, im_B_path, H, model, device = 'cuda', coarse_size=512, fine_size=None, is_scannet=False):
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
    if fine_size == coarse_size or fine_size is None:
        imgs_coarse, H_new = resize_image_with_H(imgs_ori, size=coarse_size, verbose=False, patch_size=16, H_ori=H)
        view1, view2 = imgs_coarse
        view1, view2 = collate_with_cat([(view1, view2)])
        corresps12, corresps21 = spider_symmetric_inference(model, view1, view2, device)
        warp0, certainty0 = match(corresps12)
        warp1, certainty1 = match(corresps21, inverse=True)  
        h1, w1 = imgs_coarse[0]['true_shape'][0]
        h2, w2 = imgs_coarse[1]['true_shape'][0]
    else:
        imgs_coarse, _ = resize_image_with_H(imgs_ori, size=coarse_size, H_ori=H, verbose=False)
        imgs_fine, H_new = resize_image_with_H(imgs_ori, size=fine_size, H_ori=H, verbose=False)

        view1_coarse, view2_coarse = imgs_coarse
        view1_coarse, view2_coarse = collate_with_cat([(view1_coarse, view2_coarse)])
        view1, view2 = imgs_fine
        view1, view2 = collate_with_cat([(view1, view2)])
        low_corresps12, corresps12, low_corresps21, corresps21 = spider_symmetric_inference_upsample(model, view1_coarse, view2_coarse, view1, view2, device)
        warp0, certainty0 = match_upsample(corresps12, low_corresps12)
        warp1, certainty1 = match_upsample(corresps21, low_corresps21, inverse=True)
        h1, w1 = imgs_fine[0]['true_shape'][0]
        h2, w2 = imgs_fine[1]['true_shape'][0]

    sparse_matches, mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=2048, sample_thresh=0.5)
    # kpts1, kpts2 = to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
    # return kpts1, kpts2, mconf, H_new, h1, w1, h2, w2
    return sparse_matches, mconf