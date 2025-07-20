
import os
import torch
import numpy as np
import trimesh
import copy
import spider.utils.read_write_model as rw
from spider.model import SPIDER_POINTMAP
import spider.utils.path_to_dust3r
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy, to_cpu, collate_with_cat, todevice
from dust3r.inference import make_batch_symmetric, check_if_same_size
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
import pdb
import random
import os
import shutil

from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import tqdm

def gen_rel_pose(views, norm=True):
    assert len(views) == 2
    cam1_to_w, cam2_to_w = [view['camera_pose'] for view in views]
    w_to_cam1 = np.linalg.inv(cam1_to_w)

    cam2_to_cam1 = w_to_cam1 @ cam2_to_w

    if norm: # normalize
        T = cam2_to_cam1[:3,3]
        T /= max(1e-5, np.linalg.norm(T))

    return cam2_to_cam1.astype(np.float32)

def colmap_pose(qvec, tvec):
    R_w2c = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    world2cam = np.eye(4)
    world2cam[:3, :3] = R_w2c
    world2cam[:3, 3] = tvec
    return np.linalg.inv(world2cam), world2cam


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def align_to_first_camera(camera_poses):
    """
    Align all camera poses to the first camera's coordinate frame.

    Args:
        camera_poses: Numpy/Tensor of shape (N, 4, 4) containing camera poses as SE3 transformations

    Returns:
        Numpy/Tensor of shape (N, 4, 4) containing aligned camera poses
    """
    first_cam_extrinsic_inv = closed_form_inverse_se3(camera_poses[0][None])
    aligned_poses = np.matmul(camera_poses, first_cam_extrinsic_inv)
    return aligned_poses




def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        pred_se3: Predicted SE(3) transformations
        gt_se3: Ground truth SE(3) transformations
        num_frames: Number of frames

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)
    # Compute relative camera poses between pairs
    # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
        gt_se3[pair_idx_i2]
    )
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
        pred_se3[pair_idx_i2]
    )

    # Compute the difference in rotation and translation
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg

def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)

def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram

def print_auxiliary_info(*input_views):
    view_tags = []
    for i,view in enumerate(input_views, 1):
        tags = []
        if 'camera_intrinsics' in view:
            tags.append(f'K{i}')
        if 'camera_pose' in view:
            tags.append(f'P{i}')
        if 'depthmap' in view:
            tags.append(f'D{i}')
        print(f'>> Receiving {{{"+".join(tags)}}} for view{i}')
        view_tags.append(tags)
    return view_tags

def iter_views(views, device='numpy'):
    if device:
        views = todevice(views, device)
    assert views['img'].ndim == 4
    B = len(views['img'])
    for i in range(B):
        view = {k:(v[i] if isinstance(v, (np.ndarray,torch.Tensor)) else v) for k,v in views.items()}
        yield view

def add_relpose(view, cam2_to_world, cam1_to_world=None):
    if cam2_to_world is not None:
        cam1_to_world = todevice(cam1_to_world, 'numpy')
        cam2_to_world = todevice(cam2_to_world, 'numpy')
        def fake_views(i):
            return [dict(camera_pose=np.eye(4) if cam1_to_world is None else cam1_to_world[i]), 
                    dict(camera_pose=cam2_to_world[i]) ]
        if cam2_to_world.ndim == 2:
            known_pose = gen_rel_pose(fake_views(slice(None)))
        else:
            known_pose = [gen_rel_pose(fake_views(i)) for i,v in enumerate(iter_views(view))]
            known_pose = torch.stack([todevice(k, view['img'].device) for k in known_pose])
        view['known_pose'] = known_pose

@torch.no_grad()
def inference_relpose(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        # pdb.set_trace()
        res = inference_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result

def inference_of_one_batch(batch, model, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    print_auxiliary_info(view1, view2)
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng', 'name'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    #pdb.set_trace()
    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            cam1 = view1.get('camera_pose')
            cam2 = view2.get('camera_pose')
            # pdb.set_trace()
            add_relpose(view1, cam2_to_world=cam2, cam1_to_world=cam1)
            add_relpose(view2, cam2_to_world=cam2, cam1_to_world=cam1)
            pred1, pred2 = model(view1, view2)

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2)
    return result[ret] if ret else result

def get_reconstructed_scene_with_known_pose(filelist, model, poses, outdir, device='cuda', silent=False, 
                                            min_conf_thr = 3, image_size=512, schedule='linear', niter=300, as_pointcloud=True):
    imgs = load_images(filelist, size=image_size, square_ok= True, verbose=not silent)
    for i, img in enumerate(imgs):
        img_name = filelist[i].split('/')[-1]
        if poses is not None:
            current_pose = poses.get(img_name, None)
            img['camera_pose'] = torch.from_numpy(current_pose)[None]
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    pairs = make_pairs(imgs)
    # pdb.set_trace()
    output = inference_relpose(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
    outfile = get_3D_model_from_scene(outdir, silent, scene, as_pointcloud=as_pointcloud)
    cams2world = scene.get_im_poses() #[N, 4, 4]
    world2cams = cams2world.cpu().detach().inverse()
    return world2cams


if __name__ == '__main__':
    datasets_path = '/cis/net/io96/data/zshao/JHU-ULTRA-360/JHU-ULTRA-360'
    output_path = '/cis/net/io96/data/zshao/JHU-results'

    
    
    model = SPIDER_POINTMAP.from_pretrained("/cis/home/zshao14/checkpoints/aerialdust3r_relpose_dpt512_0716/checkpoint-best.pth").to('cuda')
    # model = Spider.from_pretrained("/cis/home/zshao14/checkpoints/spider_aerialdust3r_relpose/checkpoint-best.pth").to('cuda')
    
    per_building_results = {}
    outdir_path = '/cis/net/io96/data/zshao/JHU-results/spider_dust3r_dpt512_ga_gt'
    os.makedirs(outdir_path, exist_ok=True)

    for dir_name in sorted(os.listdir(datasets_path)):
        print('Processing: ' + dir_name)
        target_folder = os.path.join('/cis/net/io96/data/zshao/JHU-sampled',  dir_name)
        sampled_imgs = sorted(os.listdir(target_folder))
        sampled_img_path = [os.path.join(target_folder, img) for img in sampled_imgs]

        output_folder = os.path.join(outdir_path, dir_name)
        os.makedirs(output_folder, exist_ok=True)

        gt_cameras, gt_images, _ = rw.read_model(os.path.join(datasets_path, dir_name, 'sparse/0'), ext='.bin')
        # est_cameras, est_images, _ = rw.read_model(os.path.join(output_path, 'spider-jhu', dir_name, 'recon_0/models/0'), ext='.bin')
        
        gt_images_id = {image.name: idx for idx, image in gt_images.items()}
        # est_images_id = {image.name.split('/')[-1]: idx for idx, image in est_images.items()}
           
        print(f"We have {len(sampled_img_path)} common images")
        gt_poses = {}
        est_poses = {}
        gt_extrinsics = []

        for name in sampled_imgs:
            gt_pose, gt_extrinsic = colmap_pose(gt_images[gt_images_id[name]].qvec, gt_images[gt_images_id[name]].tvec)
            gt_poses[name] = gt_pose
            gt_extrinsics.append(gt_extrinsic) 
            # est_pose, _ = colmap_pose(est_images[est_images_id[name]].qvec, est_images[est_images_id[name]].tvec)
            # est_poses[name] = est_pose

           
        gt_extrinsics = np.stack(gt_extrinsics, axis=0)
        gt_extrinsics = torch.from_numpy(gt_extrinsics).float()


        gt_se3 = align_to_first_camera(gt_extrinsics)
        # pred_extrinsics1 = get_reconstructed_scene_with_known_pose(sampled_img_path, model, poses=None, outdir=output_folder)
        pred_extrinsics1 = get_reconstructed_scene_with_known_pose(sampled_img_path, model, gt_poses, output_folder)
        # pred_extrinsics2 = get_reconstructed_scene_with_known_pose(sampled_img_path, model, est_poses)

        pred1_se3 = align_to_first_camera(pred_extrinsics1.cpu().detach())
        # pred2_se3 = align_to_first_camera(pred_extrinsics2.cpu().detach())

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred1_se3, gt_se3, len(sampled_imgs))


        Racc_15 = (rel_rangle_deg < 15).float().mean().item()
        Tacc_15 = (rel_tangle_deg < 15).float().mean().item()
        Racc_10 = (rel_rangle_deg < 10).float().mean().item()
        Tacc_10 = (rel_tangle_deg < 10).float().mean().item()
        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()
        Racc_3 = (rel_rangle_deg < 3).float().mean().item()
        Tacc_3 = (rel_tangle_deg < 3).float().mean().item()
        print(f"R_ACC@15: {Racc_15:.4f}")
        print(f"T_ACC@15: {Tacc_15:.4f}")

        rError = rel_rangle_deg.numpy()
        tError = rel_tangle_deg.numpy()

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

        per_building_results[dir_name] = {
            "rError": rError,
            "tError": tError,
            "Racc_15": Racc_15,
            "Tacc_15": Tacc_15,
            "Racc_10": Racc_10,
            "Tacc_10": Tacc_10,
            "Racc_3": Racc_3,
            "Tacc_3": Tacc_3,
            "Racc_5": Racc_5,
            "Tacc_5": Tacc_5,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
            
        }

        print("="*80)
        # Print results with colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(f"{BOLD}{BLUE}AUC of {dir_name} test set:{RESET} {GREEN}{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3){RESET}")
        mean_RRA_15_by_now = np.mean([per_building_results[dirname]["Racc_15"] for dirname in per_building_results])
        mean_RTA_15_by_now = np.mean([per_building_results[dirname]["Tacc_15"] for dirname in per_building_results])  
        mean_AUC_30_by_now = np.mean([per_building_results[dirname]["Auc_30"] for dirname in per_building_results])
        mean_AUC_15_by_now = np.mean([per_building_results[dirname]["Auc_15"] for dirname in per_building_results])
        mean_AUC_5_by_now = np.mean([per_building_results[dirname]["Auc_5"] for dirname in per_building_results])
        mean_AUC_3_by_now = np.mean([per_building_results[dirname]["Auc_3"] for dirname in per_building_results])
        print(f"{BOLD}{BLUE}Mean AUC of categories by now:{RESET} {RED}{mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_RRA_15_by_now:.4f} (RRA@15), {mean_RTA_15_by_now:.4f} (RTA@15){RESET}")
        print("="*80)
        # pdb.set_trace()
    # Print summary results
    print("\nSummary of AUC results:")
    print("-"*50)
    for dirname in sorted(per_building_results.keys()):
        print(f"{dirname:<15}: {per_building_results[dirname]['Auc_30']:.4f} (AUC@30), {per_building_results[dirname]['Auc_15']:.4f} (AUC@15), {per_building_results[dirname]['Auc_5']:.4f} (AUC@5), {per_building_results[dirname]['Auc_3']:.4f} (AUC@3)")
        print(f"{per_building_results[dirname]['Racc_15']:.4f} (RRA@15), {per_building_results[dirname]['Tacc_15']:.4f} (RTA@15), {per_building_results[dirname]['Racc_5']:.4f} (RRA@5), {per_building_results[dirname]['Tacc_5']:.4f} (RTA@5), {per_building_results[dirname]['Racc_3']:.4f} (RRA@3), {per_building_results[dirname]['Tacc_3']:.4f} (RTA@3)")

    if per_building_results:
        mean_RRA_15 = np.mean([per_building_results[dirname]["Racc_15"] for dirname in per_building_results])
        mean_RTA_15 = np.mean([per_building_results[dirname]["Tacc_15"] for dirname in per_building_results])  
        mean_RRA_10 = np.mean([per_building_results[dirname]["Racc_10"] for dirname in per_building_results])
        mean_RTA_10 = np.mean([per_building_results[dirname]["Tacc_10"] for dirname in per_building_results])  
        mean_RRA_5 = np.mean([per_building_results[dirname]["Racc_5"] for dirname in per_building_results])
        mean_RTA_5 = np.mean([per_building_results[dirname]["Tacc_5"] for dirname in per_building_results])  
        mean_RRA_3 = np.mean([per_building_results[dirname]["Racc_3"] for dirname in per_building_results])
        mean_RTA_3 = np.mean([per_building_results[dirname]["Tacc_3"] for dirname in per_building_results])  
        mean_AUC_30 = np.mean([per_building_results[dirname]["Auc_30"] for dirname in per_building_results])
        mean_AUC_15 = np.mean([per_building_results[dirname]["Auc_15"] for dirname in per_building_results])
        mean_AUC_5 = np.mean([per_building_results[dirname]["Auc_5"] for dirname in per_building_results])
        mean_AUC_3 = np.mean([per_building_results[dirname]["Auc_3"] for dirname in per_building_results])
        print("-"*50)
        print(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")
        print(f"{mean_RRA_15:.4f} (RRA@15), {mean_RTA_15:.4f} (RTA@15), {mean_RRA_10:.4f} (RRA@10), {mean_RTA_10:.4f} (RTA@10)")
        print(f"{mean_RRA_5:.4f} (RRA@5), {mean_RTA_5:.4f} (RTA@5), {mean_RRA_3:.4f} (RRA@3), {mean_RTA_3:.4f} (RTA@3)")

