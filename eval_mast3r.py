
import numpy as np
import os.path as osp
import os
import shutil
import torch
from tqdm import tqdm
import pdb
import json
from argparse import ArgumentParser
import random

from spider.utils.image import load_images_with_intrinsics_strict
from spider.utils.utils import compute_relative_pose, estimate_pose, compute_pose_error, compute_pose_error_T, pose_auc

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.demo import get_3D_model_from_scene, SparseGAState
from mast3r.image_pairs import make_pairs
from mast3r.inference import symmetric_inference, coarse_to_fine
from mast3r.cloud_opt.sparse_ga import extract_correspondences

import mast3r.utils.path_to_dust3r #noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import collate_with_cat

def get_reconstructed_scene(filelist, output_file_path, model, device='cuda', silent=False, image_size=512):
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    img_pairs = make_pairs(imgs, prefilter=None, symmetrize=True)
    cache_dir = os.path.join(output_file_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    try:
        scene = sparse_global_alignment(filelist, img_pairs, cache_dir,
                                    model, lr1=0.07, niter1=300, lr2=0.01, niter2=300, device=device,
                                    opt_depth=True, shared_intrinsics=False,
                                    matching_conf_thr=0)
    
        imgs = scene.imgs
        cams2world = scene.get_im_poses() #[N, 4, 4]
        world2cams = cams2world.cpu().detach().inverse()
    finally:
        shutil.rmtree(cache_dir)
    return world2cams

def load_intrinsics_and_pose(npz_path):
    camera_params = np.load(npz_path)
    K = camera_params["intrinsics"].astype(np.float32)
    T = camera_params["cam2world"].astype(np.float32)
    T_inv = np.linalg.inv(T)  
    return K, T_inv

def mast3r_match(model, device, im_A_path, im_B_path, K1_ori, K2_ori, coarse_size=512, fine_size=1344):
    imgs, _ = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=coarse_size, intrinsics=None)
    view1, view2 = imgs
    view1, view2 = collate_with_cat([(view1, view2)])
    res = symmetric_inference(model, view1, view2, device)
    descs = [r['desc'][0] for r in res]
    qonfs = [r['desc_conf'][0] for r in res]  
    # perform reciprocal matching
    corres = extract_correspondences(descs, qonfs, device=device, subsample=8)
    pts1, pts2, mconf = corres

    imgs_large, new_intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=fine_size, intrinsics=[K1_ori, K2_ori])
    h1_coarse, w1_coarse = imgs[0]['true_shape'][0]
    h2_coarse, w2_coarse = imgs[1]['true_shape'][0]

    h1, w1 = imgs_large[0]['true_shape'][0]
    h2, w2 = imgs_large[1]['true_shape'][0]
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
                                            
    kpts1, kpts2, mconf = coarse_to_fine(h1, w1, h2, w2, imgs_large, kpts1, kpts2, model, device)
    kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
    K1, K2 = new_intrinsics
    return kpts1, kpts2, K1, K2


def mast3r_match_coarse(model, device, im_A_path, im_B_path, K1_ori, K2_ori, coarse_size=512):
    imgs, new_intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=coarse_size, intrinsics=[K1_ori, K2_ori])
    view1, view2 = imgs
    view1, view2 = collate_with_cat([(view1, view2)])
    res = symmetric_inference(model, view1, view2, device)
    descs = [r['desc'][0] for r in res]
    qonfs = [r['desc_conf'][0] for r in res]  
    # perform reciprocal matching
    corres = extract_correspondences(descs, qonfs, device=device, subsample=8)
    pts1, pts2, mconf = corres                                      
    kpts1, kpts2 = pts1.cpu().numpy(), pts2.cpu().numpy()
    K1, K2 = new_intrinsics
    return kpts1, kpts2, K1, K2

def test_aerial(model, device, name, coarse_size, fine_size):
    data_root = '/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth_aerial_processed'
    with np.load(os.path.join(data_root, 'aerial_megadepth_test_scenes0015_0022.npz'), allow_pickle=True) as data:
        all_scenes = data['scenes']
        all_images = data['images']
        pairs = data['pairs']
    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    pair_inds = range(len(pairs))
    for pairind in tqdm(pair_inds):
        scene_id, idx1, idx2, score = pairs[pairind]

        scene = all_scenes[scene_id]
        seq_path = f"{data_root}/{scene}"
        im_A_name, im_B_name = all_images[idx1], all_images[idx2]
        
        ## load camera parameters
        K1_ori, T1 = load_intrinsics_and_pose(os.path.join(seq_path, im_A_name + ".npz"))
        R1, t1 = T1[:3, :3], T1[:3, 3]
        K2_ori, T2 = load_intrinsics_and_pose(os.path.join(seq_path, im_B_name + ".npz"))
        R2, t2 = T2[:3, :3], T2[:3, 3]
        R, t = compute_relative_pose(R1, t1, R2, t2)
        
        im_A_path = f"{seq_path}/{im_A_name}.jpg"
        im_B_path = f"{seq_path}/{im_B_name}.jpg"
        
        kpts1, kpts2, K1, K2 =  mast3r_match(model, device, im_A_path, im_B_path, K1_ori, K2_ori, coarse_size, fine_size)
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
    thresholds = [5, 10, 20]
    auc = pose_auc(tot_e_pose, thresholds)
    acc_5 = (tot_e_pose < 5).mean()
    acc_10 = (tot_e_pose < 10).mean()
    acc_15 = (tot_e_pose < 15).mean()
    acc_20 = (tot_e_pose < 20).mean()
    map_5 = acc_5
    map_10 = np.mean([acc_5, acc_10])
    map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
    results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    json.dump(results, open(f"./mast3r_results/aerial_{name}.json", "w"))

def test_mega(model, device, name, coarse_size, fine_size):
    data_root = '/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500'
    scene_names = [
                "0015_0.1_0.3.npz",
                "0015_0.3_0.5.npz",
                "0022_0.1_0.3.npz",
                "0022_0.3_0.5.npz",
                "0022_0.5_0.7.npz",
            ]
    
    scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in scene_names
        ]

    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    for scene_ind in range(len(scenes)):
        scene_name = os.path.splitext(scene_names[scene_ind])[0]
        scene = scenes[scene_ind]
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
            
            im_A_path = f"{data_root}/{im_paths[idx1]}"
            im_B_path = f"{data_root}/{im_paths[idx2]}"          

            kpts1, kpts2, K1, K2 =  mast3r_match(model, device, im_A_path, im_B_path, K1_ori, K2_ori, coarse_size, fine_size)
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
    thresholds = [5, 10, 20]
    auc = pose_auc(tot_e_pose, thresholds)
    acc_5 = (tot_e_pose < 5).mean()
    acc_10 = (tot_e_pose < 10).mean()
    acc_15 = (tot_e_pose < 15).mean()
    acc_20 = (tot_e_pose < 20).mean()
    map_5 = acc_5
    map_10 = np.mean([acc_5, acc_10])
    map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
    results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    json.dump(results, open(f"./mast3r_results/mega_{name}.json", "w"))

def test_scannet(model, device, name, coarse_size, fine_size):
    data_root = '/cis/net/r24a/data/zshao/data/scannet1500'
    tmp = np.load(osp.join(data_root, "test.npz"))
    pairs, rel_pose = tmp["name"], tmp["rel_pose"]
    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    pair_inds = np.random.choice(
        range(len(pairs)), size=len(pairs), replace=False
    )
    for pairind in tqdm(pair_inds, position=0, leave=True, desc="Processing pairs"):
        scene = pairs[pairind]
        scene_name = f"scene0{scene[0]}_00"
        im_A_path = osp.join(
                data_root,
                "scannet_test_1500",
                scene_name,
                "color",
                f"{scene[2]}.jpg",
            )
        im_B_path = osp.join(
                data_root,
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
                        data_root,
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
        kpts1, kpts2, K1, K2 =  mast3r_match(model, device, im_A_path, im_B_path, K1_ori, K2_ori, coarse_size, fine_size)
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
    results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    json.dump(results, open(f"./mast3r_results/scannet_{name}.json", "w"))



def test_aerial_coarse(model, device, name, coarse_size):
    data_root = '/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth_aerial_processed'
    with np.load(os.path.join(data_root, 'aerial_megadepth_test_scenes0015_0022.npz'), allow_pickle=True) as data:
        all_scenes = data['scenes']
        all_images = data['images']
        pairs = data['pairs']
    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    pair_inds = range(len(pairs))
    for pairind in tqdm(pair_inds):
        scene_id, idx1, idx2, score = pairs[pairind]

        scene = all_scenes[scene_id]
        seq_path = f"{data_root}/{scene}"
        im_A_name, im_B_name = all_images[idx1], all_images[idx2]
        
        ## load camera parameters
        K1_ori, T1 = load_intrinsics_and_pose(os.path.join(seq_path, im_A_name + ".npz"))
        R1, t1 = T1[:3, :3], T1[:3, 3]
        K2_ori, T2 = load_intrinsics_and_pose(os.path.join(seq_path, im_B_name + ".npz"))
        R2, t2 = T2[:3, :3], T2[:3, 3]
        R, t = compute_relative_pose(R1, t1, R2, t2)
        
        im_A_path = f"{seq_path}/{im_A_name}.jpg"
        im_B_path = f"{seq_path}/{im_B_name}.jpg"
        
        kpts1, kpts2, K1, K2 =  mast3r_match_coarse(model, device, im_A_path, im_B_path, K1_ori, K2_ori, coarse_size)
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
    thresholds = [5, 10, 20]
    auc = pose_auc(tot_e_pose, thresholds)
    acc_5 = (tot_e_pose < 5).mean()
    acc_10 = (tot_e_pose < 10).mean()
    acc_15 = (tot_e_pose < 15).mean()
    acc_20 = (tot_e_pose < 20).mean()
    map_5 = acc_5
    map_10 = np.mean([acc_5, acc_10])
    map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
    results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    json.dump(results, open(f"./mast3r_results/aerial_{name}.json", "w"))

def test_mega_coarse(model, device, name, coarse_size):
    data_root = '/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500'
    scene_names = [
                "0015_0.1_0.3.npz",
                "0015_0.3_0.5.npz",
                "0022_0.1_0.3.npz",
                "0022_0.3_0.5.npz",
                "0022_0.5_0.7.npz",
            ]
    
    scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in scene_names
        ]

    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    for scene_ind in range(len(scenes)):
        scene_name = os.path.splitext(scene_names[scene_ind])[0]
        scene = scenes[scene_ind]
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
            
            im_A_path = f"{data_root}/{im_paths[idx1]}"
            im_B_path = f"{data_root}/{im_paths[idx2]}"          

            kpts1, kpts2, K1, K2 =  mast3r_match_coarse(model, device, im_A_path, im_B_path, K1_ori, K2_ori, coarse_size)
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
    thresholds = [5, 10, 20]
    auc = pose_auc(tot_e_pose, thresholds)
    acc_5 = (tot_e_pose < 5).mean()
    acc_10 = (tot_e_pose < 10).mean()
    acc_15 = (tot_e_pose < 15).mean()
    acc_20 = (tot_e_pose < 20).mean()
    map_5 = acc_5
    map_10 = np.mean([acc_5, acc_10])
    map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
    results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    json.dump(results, open(f"./mast3r_results/mega_{name}.json", "w"))

def test_scannet_coarse(model, device, name, coarse_size):
    data_root = '/cis/net/r24a/data/zshao/data/scannet1500'
    tmp = np.load(osp.join(data_root, "test.npz"))
    pairs, rel_pose = tmp["name"], tmp["rel_pose"]
    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    pair_inds = np.random.choice(
        range(len(pairs)), size=len(pairs), replace=False
    )
    for pairind in tqdm(pair_inds, position=0, leave=True, desc="Processing pairs"):
        scene = pairs[pairind]
        scene_name = f"scene0{scene[0]}_00"
        im_A_path = osp.join(
                data_root,
                "scannet_test_1500",
                scene_name,
                "color",
                f"{scene[2]}.jpg",
            )
        im_B_path = osp.join(
                data_root,
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
                        data_root,
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
        kpts1, kpts2, K1, K2 =  mast3r_match_coarse(model, device, im_A_path, im_B_path, K1_ori, K2_ori, coarse_size)
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
    results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    json.dump(results, open(f"./mast3r_results/scannet_{name}.json", "w"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='mast3r', type=str)
    parser.add_argument("--dataset", default='aerial', type=str)
    parser.add_argument("--model_name", default='mast3r', type=str)
    parser.add_argument("--coarse_size", default=512, type=int)
    parser.add_argument("--fine_size", default=1024, type=int)
    args, _ = parser.parse_known_args()
    model_dict = {'mast3r': 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric', 
     'aerial-mast3r': '/cis/home/zshao14/checkpoints/checkpoint-aerial-mast3r.pth'}

    device = 'cuda'
    model = AsymmetricMASt3R.from_pretrained(model_dict[args.model_name]).to(device)
    experiment_name = args.exp_name

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic
    torch.backends.cudnn.benchmark = False  
    if args.dataset == 'scannet':
        if '512' in experiment_name:
            test_scannet_coarse(model, device, experiment_name, args.coarse_size) 
        else:
            test_scannet(model, device, experiment_name, args.coarse_size, args.fine_size) 
    elif args.dataset == 'mega1500':
        if '512' in experiment_name:
            test_mega_coarse(model, device, experiment_name, args.coarse_size)
        else:
            test_mega(model, device, experiment_name, args.coarse_size, args.fine_size)
    elif args.dataset == 'aerial':
        if '512' in experiment_name:
            test_aerial_coarse(model, device, experiment_name, args.coarses_ize)
        else:
            test_aerial(model, device, experiment_name, args.coarse_size, args.fine_size)
    