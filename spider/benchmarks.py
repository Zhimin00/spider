import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import cv2
import pdb
from spider.utils.utils import compute_relative_pose, estimate_pose, compute_pose_error, pose_auc, match, match_upsample, sample_symmetric, to_pixel_coordinates
from spider.utils.image import load_original_images, resize_image_with_intrinsics
from spider.inference import two_symmetric_inference, two_symmetric_inference_upsample
from spider.inference import coarse_to_fine as two_coarse_to_fine

from PIL import Image
import spider.utils.path_to_dust3r # noqa
from dust3r_visloc.datasets.utils import get_HW_resolution
from dust3r.utils.geometry import geotrf
from mast3r.cloud_opt.sparse_ga import extract_correspondences

import mast3r.utils.path_to_dust3r #noqa
from dust3r.utils.device import collate_with_cat
import cv2

def spider_two_match_path(im_A_path, im_B_path, K1_ori, K2_ori, model, device = 'cuda', coarse_size=512, fine_size=None):
    imgs_ori = load_original_images([im_A_path, im_B_path], verbose=False)
        
    if fine_size == coarse_size or fine_size is None:
        imgs_coarse, intrinsics = resize_image_with_intrinsics(imgs_ori, size=coarse_size, intrinsics=[K1_ori, K2_ori], verbose=False)
        K1, K2 = intrinsics
        view1, view2 = imgs_coarse
        view1, view2 = collate_with_cat([(view1, view2)])
        
        corresps, res = two_symmetric_inference(model, view1, view2, 'cuda')
        corresps12, corresps21 = corresps
        descs = [r['desc'][0] for r in res]
        qonfs = [r['desc_conf'][0] for r in res]  
        # perform reciprocal matching
        corres = extract_correspondences(descs, qonfs, device='cuda', subsample=16)
        fm_kpts0, fm_kpts1, fm_mconf = corres                                      

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

        corresps, res = two_symmetric_inference_upsample(model, view1_coarse, view2_coarse, view1, view2, 'cuda')
        low_corresps12, corresps12, low_corresps21, corresps21 = corresps
        descs = [r['desc'][0] for r in res]
        qonfs = [r['desc_conf'][0] for r in res]  
        # perform reciprocal matching
        corres = extract_correspondences(descs, qonfs, device='cuda', subsample=8)
        pts1, pts2, mconf = corres
        h1_coarse, w1_coarse = imgs_coarse[0]['true_shape'][0]
        h2_coarse, w2_coarse = imgs_coarse[1]['true_shape'][0]

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
                                                
        fm_kpts0, fm_kpts1, fm_mconf = two_coarse_to_fine(h1, w1, h2, w2, imgs_fine, kpts1, kpts2, mconf, model, 'cuda')#, max_batch_size=48)
        warp0, certainty0 = match_upsample(corresps12, low_corresps12)
        warp1, certainty1 = match_upsample(corresps21, low_corresps21, inverse=True)

        h1, w1 = imgs_fine[0]['true_shape'][0]
        h2, w2 = imgs_fine[1]['true_shape'][0]
    nMast3r_matches = fm_kpts0.size()[0]
    # print('nMast3r_matches =', nMast3r_matches)
    value = int(nMast3r_matches / 1)  
    # MAX_LIMIT = 20_000
    # value = max(100, min(value, MAX_LIMIT))
    value=5000
    sparse_matches, spider_mconf = sample_symmetric(warp0, certainty0, warp1, certainty1, num=value)#, sample_thresh=0.3)#num=10000)#, sample_mode = "balanced")
    spider_kpts0, spider_kpts1 = to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
    
    kpts0 = torch.cat([fm_kpts0, spider_kpts0], dim=0)    
    kpts1 = torch.cat([fm_kpts1, spider_kpts1], dim=0)
    mconf = torch.cat([fm_mconf, spider_mconf], dim=0)
    return kpts0, kpts1, mconf, K1, K2


def load_intrinsics_and_pose(npz_path):
    camera_params = np.load(npz_path)
    K = camera_params["intrinsics"].astype(np.float32)
    T = camera_params["cam2world"].astype(np.float32)
    T_inv = np.linalg.inv(T)  
    return K, T_inv

class AerialMegaDepthPoseEstimationBenchmark:
    def __init__(self, data_root="data/megadepth") -> None:
        self.data_root = data_root #'aerial-ground-test-0.1.npz aerial_megadepth_test_scenes0015_0022.npz
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

    def benchmark_two(self, model, device='cuda', model_name = 'msfr-warp-concat', debug=False, coarse_size=512, fine_size=1344):
        # output_file = f'/cis/home/zshao14/Downloads/spider/output/{model_name}_aerial.txt'
        
        with torch.no_grad():
            data_root = self.data_root
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            thresholds = [5, 10, 20]

            pair_inds = range(300)#len(self.pairs))
            # with open(output_file, 'w') as f:
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

                kpts1, kpts2, mconf, K1, K2 = spider_two_match_path(im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size)
                
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

    def benchmark_two(self, model, device='cuda', model_name = 'spider-two', coarse_size=512, fine_size=1344):
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
                    kpts1, kpts2, mconf, K1, K2 = spider_two_match_path(im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size)
                    
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
            print(f"{model_name} megadepth auc: {auc}")
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

    def benchmark_two(self, two_model, device='cuda', model_name = 'spider-two', coarse_size=512, fine_size=1344):
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
                
                kpts1, kpts2, mconf, K1, K2 = spider_two_match_path(im_A_path, im_B_path, K1_ori, K2_ori, two_model, device = 'cuda', coarse_size=coarse_size, fine_size=fine_size)
                kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                
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
