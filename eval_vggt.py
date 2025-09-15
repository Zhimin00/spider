
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
from spider.utils.utils import compute_relative_pose, estimate_pose, compute_pose_error, compute_pose_error_T, pose_auc

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_withK, load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import sys
aliked_path = os.path.abspath("/cis/home/zshao14/Downloads/ALIKED")
sys.path.insert(0, aliked_path)
from nets.aliked import ALIKED
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])
def tensor_to_pil(x, unnormalize=False):
    if unnormalize:
        x = x * (imagenet_std[:, None, None].to(x.device)) + (imagenet_mean[:, None, None].to(x.device))
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)

def tensor_to_plt(x, unnormalize=False):
    if unnormalize:
        x = x * (imagenet_std[:, None, None].to(x.device)) + (imagenet_mean[:, None, None].to(x.device))
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return x

def plot_image_kpts(im1, im2, kpts1, kpts2, is_valid):
    im1 = tensor_to_plt(im1)
    im2 = tensor_to_plt(im2)
    H1, W1, H2, W2 = *im1.shape[:2], *im2.shape[:2]
    img1 = np.pad(im1, ((0, max(H2 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img2 = np.pad(im2, ((0, max(H1 - H2, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img1, img2), axis=1)


    plt.figure()
    plt.imshow(img)
    plt.axis('off')  
    plt.savefig('/cis/home/zshao14/Documents/vggt/scannet/raw.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    num_matches = len(kpts1)
    print(num_matches)

    plt.figure()
    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(num_matches):
        (x0, y0), (x1, y1) = kpts1[i].T, kpts2[i].T
        # pl.plot([x0, x1 + W0], [y0, y1], color=cmap(i / (n_viz - 1)), scalex=False, scaley=False, linewidth=0.3)
        plt.scatter(x0, y0, color=cmap(i / (num_matches - 1)), s=0.1)
        plt.scatter(x1 + W1, y1, color=cmap(i / (num_matches - 1)), s=0.1)
        if is_valid[i]:
            plt.plot([x0, x1 + W1], [y0, y1], color=cmap(i / (num_matches - 1)), scalex=False, scaley=False, linewidth=0.3)
    plt.axis('off') 
    # pl.show(block=True)
    plt.savefig('/cis/home/zshao14/Documents/vggt/scannet/vggt-dot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plt.figure()
    # plt.imshow(img)
    # cmap = plt.get_cmap('jet')
    # for i in range(len(kpts1)):
    #     (x0, y0), (x1, y1) = kpts1[i].T, kpts2[i].T
    #     plt.plot([x0, x1 + W1], [y0, y1], color=cmap(i / (num_matches - 1)), scalex=False, scaley=False, linewidth=0.3)
    # plt.axis('off') 
    # # pl.show(block=True)
    # plt.savefig('/cis/home/zshao14/Documents/vggt/scannet/vggt_matches.png', dpi=300, bbox_inches='tight')
    # plt.close()
def save_pca_image(tensor, dir_path, filename="pca_image.png", index=0):
    """
    Reduces C → 3 using PCA on tensor[index] and saves it as an image.
    
    Args:
        tensor (torch.Tensor): shape (B, C, H, W)
        dir_path (str): directory to save the image
        filename (str): name of the output file
        index (int): index in the batch to visualize
    """
    os.makedirs(dir_path, exist_ok=True)

    # Select image: shape (C, H, W)
    X = tensor[index]
    C, H, W = X.shape

    # Flatten for PCA: shape (H*W, C)
    X_flat = X.view(C, -1).T  # (H*W, C)

    # Apply PCA to reduce to 3 channels
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_flat.cpu().numpy())  # (H*W, 3)

    # Reshape back to (3, H, W)
    X_pca_image = X_pca.T.reshape(3, H, W)

    # Normalize to [0, 255] for saving as uint8
    X_img = X_pca_image - X_pca_image.min()
    X_img = X_img / X_img.max()
    X_img = (X_img * 255).astype(np.uint8)  # (3, H, W)

    # Convert to (H, W, 3) for saving
    X_img = np.transpose(X_img, (1, 2, 0))

    # Save using matplotlib
    save_path = os.path.join(dir_path, filename)
    plt.imsave(save_path, X_img)

    print(f"Saved PCA image to: {save_path}")

def load_intrinsics_and_pose(npz_path):
    camera_params = np.load(npz_path)
    K = camera_params["intrinsics"].astype(np.float32)
    T = camera_params["cam2world"].astype(np.float32)
    T_inv = np.linalg.inv(T)  
    return K, T_inv


def test_aerial(model, device, name):
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
        images = load_and_preprocess_images([im_A_path, im_B_path], mode='pad').to(device)
        # ipdb.set_trace()
        images = images[None]  # add batch dimension
        with torch.no_grad():
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        R1_est = extrinsic[0][0][:3,:3]
        t1_est =extrinsic[0][0][:3, 3]
        R2_est = extrinsic[0][1][:3,:3]
        t2_est =extrinsic[0][1][:3, 3]
        R1_est, R2_est, t1_est, t2_est = R1_est.cpu().numpy(), R2_est.cpu().numpy(), t1_est.cpu().numpy(), t2_est.cpu().numpy()
        R_est, t_est = compute_relative_pose(R1_est, t1_est, R2_est, t2_est)
        T1_to_2_est = np.concatenate((R_est, t_est[:, None]), axis=-1)  #
        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
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
    print(f"{name} auc: {auc}")
    results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    json.dump(results, open(f"./mast3r_results/aerial_{name}.json", "w"))

def test_mega(model, device, name):
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
            images = load_and_preprocess_images([im_A_path, im_B_path], mode='pad')
            images = images[None].to(device)  # add batch dimension
            with torch.no_grad():
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            R1_est = extrinsic[0][0][:3,:3]
            t1_est =extrinsic[0][0][:3, 3]
            R2_est = extrinsic[0][1][:3,:3]
            t2_est =extrinsic[0][1][:3, 3]
            R1_est, R2_est, t1_est, t2_est = R1_est.cpu().numpy(), R2_est.cpu().numpy(), t1_est.cpu().numpy(), t2_est.cpu().numpy()
            R_est, t_est = compute_relative_pose(R1_est, t1_est, R2_est, t2_est)
            T1_to_2_est = np.concatenate((R_est, t_est[:, None]), axis=-1)  #
            e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
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

def test_scannet(model, device, name):
    data_root = '/cis/net/r24a/data/zshao/data/scannet1500'
    tmp = np.load(osp.join(data_root, "test.npz"))
    pairs, rel_pose = tmp["name"], tmp["rel_pose"]
    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    pair_inds = np.random.choice(
        range(len(pairs)), size=len(pairs), replace=False
    )
    aliked_model = ALIKED(model_name='aliked-n32', top_k=1024)
    for pairind in tqdm(pair_inds, position=0, leave=True, desc="Processing pairs"):
        if pairind != 888:
            continue
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
        images, K = load_and_preprocess_images_withK([im_A_path, im_B_path], [K1_ori, K2_ori], mode='pad')

        im1 = images[0].to(device)
        im2 = images[1].to(device)
        h1_new, w1_new = im1.shape[-2:]
       
        images = images[None].to(device)  # add batch dimension
        with torch.no_grad():
            ori_kpts1 = aliked_model.forward(im1[None])['keypoints'][0]
            wh1 = torch.tensor([w1_new - 1, h1_new - 1],device=ori_kpts1.device)
            ori_kpts1 = wh1 * (ori_kpts1 + 1) / 2

            aggregated_tokens_list, ps_idx = model.aggregator(images)
            track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=ori_kpts1[None])
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        ori_kpts2 = track_list[-1][0][1]
        kpts1, kpts2, vis_score, conf_score = ori_kpts1.cpu().numpy(), ori_kpts2.cpu().numpy(), vis_score[0][1].cpu().numpy(), conf_score[0][1].cpu().numpy()
        
        # is_valid = (conf_score > 0.01) & (vis_score > 0.1)
        is_valid = (kpts1[:,0] < 100) & (vis_score > 0.1)
        # kpts1 = kpts1[is_valid]
        # kpts2 = kpts2[is_valid]

        plot_image_kpts(im1, im2, kpts1, kpts2, is_valid)
        R1_est = extrinsic[0][0][:3,:3]
        t1_est =extrinsic[0][0][:3, 3]
        R2_est = extrinsic[0][1][:3,:3]
        t2_est =extrinsic[0][1][:3, 3]
        R1_est, R2_est, t1_est, t2_est = R1_est.cpu().numpy(), R2_est.cpu().numpy(), t1_est.cpu().numpy(), t2_est.cpu().numpy()
        R_est, t_est = compute_relative_pose(R1_est, t1_est, R2_est, t2_est)
        T1_to_2_est = np.concatenate((R_est, t_est[:, None]), axis=-1)  #
        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)

        e_pose = max(e_t, e_R)
        print(e_pose)
        K1, K2 = K
        # for _ in range(5):
        #     shuffling = np.random.permutation(np.arange(len(kpts1)))
        #     kpts1 = kpts1[shuffling]
        #     kpts2 = kpts2[shuffling]
        try:
            norm_threshold = 0.5 / (
            np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
            R_est, t_est, mask = estimate_pose(kpts1, kpts2, K1, K2, norm_threshold, conf=0.99999,)
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
        print(e_pose)
        pdb.set_trace()
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
    print(auc)
    results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    json.dump(results, open(f"./mast3r_results/scannet_{name}.json", "w"))


def test_jhu(model, device, name):
    # detector = dad_detector.load_DaD()

    data_root = '/cis/net/io96/data/zshao/JHU-ULTRA-360/pairs'
    scene_names = [
                "10_AMES.npy",
                "24_Clark.npy",
                "78_Shriver.npy",
            ]
    
    scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in scene_names
        ]

    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    for scene_ind in range(len(scenes)):
        scene_name = scene_names[scene_ind].split('.')[0]
        pairs = scenes[scene_ind]
        
        pair_inds = range(len(pairs))
        for pairind in tqdm(pair_inds):
            im1_name, im2_name,  shared_points, overlap_ratio, T1, T2, K1, K2= pairs[pairind]
            K1_ori = np.array(K1).reshape(3, 3)
            K2_ori = np.array(K2).reshape(3, 3)
            T1 = np.array(T1).reshape(4, 4)
            T2 = np.array(T2).reshape(4, 4)
            
            R1, t1 = T1[:3, :3], T1[:3, 3]
            R2, t2 = T2[:3, :3], T2[:3, 3]
            R, t = compute_relative_pose(R1, t1, R2, t2)
            
            im_A_path = f"{data_root}/{scene_name}/{im1_name}"
            im_B_path = f"{data_root}/{scene_name}/{im2_name}"          
            images = load_and_preprocess_images([im_A_path, im_B_path], mode='pad')
            images = images[None].to(device)  # add batch dimension
            with torch.no_grad():
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            R1_est = extrinsic[0][0][:3,:3]
            t1_est = extrinsic[0][0][:3, 3]
            R2_est = extrinsic[0][1][:3,:3]
            t2_est =extrinsic[0][1][:3, 3]
            R1_est, R2_est, t1_est, t2_est = R1_est.cpu().numpy(), R2_est.cpu().numpy(), t1_est.cpu().numpy(), t2_est.cpu().numpy()
            R_est, t_est = compute_relative_pose(R1_est, t1_est, R2_est, t2_est)
            T1_to_2_est = np.concatenate((R_est, t_est[:, None]), axis=-1)  #
            e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
            e_pose = max(e_t, e_R)
            tot_e_t.append(e_t)
            tot_e_R.append(e_R)
            tot_e_pose.append(e_pose)
    tot_e_pose = np.array(tot_e_pose)
    thresholds = [5, 10, 20, 30]
    auc = pose_auc(tot_e_pose, thresholds)
    acc_5 = (tot_e_pose < 5).mean()
    acc_10 = (tot_e_pose < 10).mean()
    acc_15 = (tot_e_pose < 15).mean()
    acc_20 = (tot_e_pose < 20).mean()
    acc_30 = (tot_e_pose < 30).mean()
    map_5 = acc_5
    map_10 = np.mean([acc_5, acc_10])
    map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
    results = { "auc_30": auc[3],
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
    print(auc)
    json.dump(results, open(f"./mast3r_results/jhu_{name}.json", "w"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='vggt', type=str)
    parser.add_argument("--dataset", default='aerial', type=str)
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # model = VGGT()
    # model.load_state_dict(torch.load("/cis/home/zshao14/checkpoints/model_tracker_fixed_e20.pt"))
    # model = model.to(device)

    experiment_name = args.exp_name

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic
    torch.backends.cudnn.benchmark = False  
    if args.dataset == 'scannet':
        test_scannet(model, device, experiment_name) 
    elif args.dataset == 'mega1500':
        test_mega(model, device, experiment_name)
    elif args.dataset == 'aerial':
        test_aerial(model, device, experiment_name)
    elif args.dataset == 'jhu':
        test_jhu(model, device, experiment_name)
    