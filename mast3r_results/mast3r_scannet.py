import numpy as np
import os.path as osp
import os
import shutil
import torch
from tqdm import tqdm
import pdb

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

# for pairind in tqdm(pair_inds, position=0, leave=True, desc="Processing pairs"):
#         scene = pairs[pairind]
#         scene_name = f"scene0{scene[0]}_00"
#         im_A_path = osp.join(
#                 data_root,
#                 "scannet_test_1500",
#                 scene_name,
#                 "color",
#                 f"{scene[2]}.jpg",
#             )
#         im_B_path = osp.join(
#                 data_root,
#                 "scannet_test_1500",
#                 scene_name,
#                 "color",
#                 f"{scene[3]}.jpg",
#             )
#         T_gt = rel_pose[pairind].reshape(3, 4)
#         R, t = T_gt[:3, :3], T_gt[:3, 3]
#         T1_est, T2_est = get_reconstructed_scene([im_A_path, im_B_path], output_file_path, model, silent=True)
#         e_t, e_R = compute_pose_error(T1_est, T2_est, R, t)
#         e_pose = max(e_t, e_R)        


if __name__ == '__main__':
    device = 'cuda'
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    data_root = '/cis/net/r24a/data/zshao/data/scannet1500'
    output_file_path = '/cis/net/r24a/data/zshao/data/scannet1500_result'
    
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
                        "scans_test",
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
        imgs, _ = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=512, intrinsics=None)
        view1, view2 = imgs
        view1, view2 = collate_with_cat([(view1, view2)])
        res = symmetric_inference(model, view1, view2, device)
        descs = [r['desc'][0] for r in res]
        qonfs = [r['desc_conf'][0] for r in res]  
        # perform reciprocal matching
        corres = extract_correspondences(descs, qonfs, device=device, subsample=8)
        pts1, pts2, mconf = corres

        imgs_large, new_intrinsics = load_images_with_intrinsics_strict([im_A_path, im_B_path], size=1344, intrinsics=[K1_ori, K2_ori])
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
                                                
        kpts1, kpts2 = coarse_to_fine(h1, w1, h2, w2, imgs_large, kpts1, kpts2, model, device)
        kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
        K1, K2 = new_intrinsics        
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
    thresholds = [5, 10, 20]
    auc = pose_auc(tot_e_pose, thresholds)
    acc_5 = (tot_e_pose < 5).mean()
    acc_10 = (tot_e_pose < 10).mean()
    acc_15 = (tot_e_pose < 15).mean()
    acc_20 = (tot_e_pose < 20).mean()
    map_5 = acc_5
    map_10 = np.mean([acc_5, acc_10])
    map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
    print(f"'auc_5': {auc[0]}, 'auc_10': {auc[1]}, 'auc_20': {auc[2]}, 'map_5': {map_5}, 'map_10': {map_10}, 'map_20': {map_20}")