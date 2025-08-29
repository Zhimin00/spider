import warnings
import numpy as np
import cv2
import math
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import spider.utils.path_to_dust3r
from dust3r.utils.device import todevice

import kornia
import pdb
from typing import Optional, Union
Device = Union[str, torch.device]

def gen_rel_pose(views, norm=True):
    assert len(views) == 2
    cam1_to_w, cam2_to_w = [view['camera_pose'] for view in views]
    w_to_cam1 = np.linalg.inv(cam1_to_w)

    cam2_to_cam1 = w_to_cam1 @ cam2_to_w

    if norm: # normalize
        T = cam2_to_cam1[:3,3]
        T /= max(1e-5, np.linalg.norm(T))

    return cam2_to_cam1.astype(np.float32)


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

def make_symmetric_pairs(views):
    pairs = []
    pairs.append((views[0], views[1]))
    pairs.append((views[1], views[0]))
    return pairs

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


def tensor_to_pil(x, unnormalize=False):
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def match_single(im_A_to_im_B, certainty, certainty_s16, attenuate_cert=True, inverse=False, batched=False):
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
    low_res_certainty = F.interpolate(
                certainty_s16, size=(h, w), align_corners=False, mode="bilinear"
            )
    cert_clamp = 0
    factor = 0.5
    low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)
    certainty = certainty - (low_res_certainty if attenuate_cert else 0)
    im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
    device = im_A_to_im_B.device
    # Create im_A meshgrid
    im_A_coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
        ),
        indexing='ij'
    )
    im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
    im_A_coords = im_A_coords[None].expand(b, 2, h, w)
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

def match(corresps, attenuate_cert=True, inverse=False):
    finest_scale = 1
    im_A_to_im_Bs = corresps[finest_scale]["flow"] 
    certaintys = corresps[finest_scale]["certainty"]
    certainty_s16s= corresps[16]["certainty"]
    warp, certainty = match_single(im_A_to_im_B=im_A_to_im_Bs, certainty=certaintys, certainty_s16=certainty_s16s, attenuate_cert=attenuate_cert, inverse=inverse)
    return warp, certainty

def match_upsample(corresps, low_corresps, attenuate_cert=True, inverse=False):
    finest_scale = 1
    im_A_to_im_Bs = corresps[finest_scale]["flow"] 
    certaintys = corresps[finest_scale]["certainty"]
    certainty_s16s= low_corresps[16]["certainty"]
    warp, certainty = match_single(im_A_to_im_B=im_A_to_im_Bs, certainty=certaintys, certainty_s16=certainty_s16s, attenuate_cert=attenuate_cert, inverse=inverse)
    return warp, certainty

def fmwarp_match(corresps, attenuate_cert=True, inverse=False):
    im_A_to_im_Bs = corresps["flow"] 
    certaintys = corresps["certainty"]
    certainty_s16s= corresps["gm_certainty"]
    warp, certainty = match_single(im_A_to_im_B=im_A_to_im_Bs, certainty=certaintys, certainty_s16=certainty_s16s, attenuate_cert=attenuate_cert, inverse=inverse)
    return warp, certainty

def match_symmetric(corresps, attenuate_cert=True):
    finest_scale = 1
    im_A_to_im_Bs = corresps[finest_scale]["flow"] 
    certaintys = corresps[finest_scale]["certainty"]
    certainty_s16s= corresps[16]["certainty"]

    b = len(im_A_to_im_Bs)
    assert b == 2, "Non-symmetric pairs"
    warp1, certainty1 = match_single(im_A_to_im_B=im_A_to_im_Bs[0], certainty=certaintys[0], certainty_s16=certainty_s16s[0], attenuate_cert=attenuate_cert)
    warp2, certainty2 = match_single(im_A_to_im_B=im_A_to_im_Bs[1], certainty=certaintys[1], certainty_s16=certainty_s16s[1], attenuate_cert=attenuate_cert, inverse=True)
    return warp1, certainty1, warp2, certainty2

def match_symmetric_upsample(corresps, low_corresps, attenuate_cert=True):
    finest_scale = 1
    im_A_to_im_Bs = corresps[finest_scale]["flow"] 
    certaintys = corresps[finest_scale]["certainty"]
    certainty_s16s= low_corresps[16]["certainty"]

    b = len(im_A_to_im_Bs)
    assert b == 2, "Non-symmetric pairs"
    warp1, certainty1 = match_single(im_A_to_im_B=im_A_to_im_Bs[0], certainty=certaintys[0], certainty_s16=certainty_s16s[0], attenuate_cert=attenuate_cert)
    warp2, certainty2 = match_single(im_A_to_im_B=im_A_to_im_Bs[1], certainty=certaintys[1], certainty_s16=certainty_s16s[1], attenuate_cert=attenuate_cert, inverse=True)
    return warp1, certainty1, warp2, certainty2

def sample_symmetric(matches1, certainty1, matches2, certainty2, num=10000, sample_mode = "threshold_balanced", sample_thresh=0.05):
    if "threshold" in sample_mode:
        upper_thresh = sample_thresh
        certainty1 = certainty1.clone()
        certainty1[certainty1 > upper_thresh] = 1
        certainty2 = certainty2.clone()
        certainty2[certainty2 > upper_thresh] = 1
    matches1, certainty1 = (
        matches1.reshape(-1, 4),
        certainty1.reshape(-1),
    )
    matches2, certainty2 = (
        matches2.reshape(-1, 4),
        certainty2.reshape(-1),
    )
    matches = torch.cat([matches1, matches2], dim=0)
    certainty = torch.cat([certainty1, certainty2], dim=0)
    expansion_factor = 4 if "balanced" in sample_mode else 1
    good_samples = torch.multinomial(certainty, 
                        num_samples = min(expansion_factor*num, len(certainty)), 
                        replacement=False)
    good_matches, good_certainty = matches[good_samples], certainty[good_samples]
    if "balanced" not in sample_mode:
        return good_matches, good_certainty
    density = kde(good_matches, std=0.1)
    p = 1 / (density+1)
    p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
    balanced_samples = torch.multinomial(p, 
                        num_samples = min(num,len(good_certainty)), 
                        replacement=False)
    return good_matches[balanced_samples], good_certainty[balanced_samples]


def sample(matches, certainty, num=10000, sample_mode = "threshold_balanced", sample_thresh=0.05):
    if "threshold" in sample_mode:
        upper_thresh = sample_thresh
        certainty = certainty.clone()
        certainty[certainty > upper_thresh] = 1
    matches, certainty = (
        matches.reshape(-1, 4),
        certainty.reshape(-1),
    )
    expansion_factor = 4 if "balanced" in sample_mode else 1
    good_samples = torch.multinomial(certainty, 
                        num_samples = min(expansion_factor*num, len(certainty)), 
                        replacement=False)
    good_matches, good_certainty = matches[good_samples], certainty[good_samples]
    if "balanced" not in sample_mode:
        return good_matches, good_certainty
    density = kde(good_matches, std=0.1)
    p = 1 / (density+1)
    p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
    balanced_samples = torch.multinomial(p, 
                        num_samples = min(num,len(good_certainty)), 
                        replacement=False)
    return good_matches[balanced_samples], good_certainty[balanced_samples]

def sample_batched(matches, certainty, num=10000, sample_mode="threshold_balanced", sample_thresh=0.05):
    B, N, _ = matches.shape
    batched_matches = []
    batched_certainty = []
    for b in range(B):
        good_matches_b, good_certainty_b = sample(matches[b], certainty[b], num=num, sample_mode=sample_mode, sample_thresh=sample_thresh)
        batched_matches.append(good_matches_b)
        batched_certainty.append(good_certainty_b)
    # Stack the results to return tensors of shape [B, num, 4] and [B, num]
    return torch.stack(batched_matches), torch.stack(batched_certainty)



def kde(x, std = 0.1, half = False, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

def to_pixel_coordinates( coords, H_A, W_A, H_B = None, W_B = None):
    if coords.shape[-1] == 2:
        return _to_pixel_coordinates(coords, H_A, W_A) 
    
    if isinstance(coords, (list, tuple)):
        kpts_A, kpts_B = coords[0], coords[1]
    else:
        kpts_A, kpts_B = coords[...,:2], coords[...,2:]
    return _to_pixel_coordinates(kpts_A, H_A, W_A), _to_pixel_coordinates(kpts_B, H_B, W_B)

def _to_pixel_coordinates( coords, H, W):
    kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
    return kpts
 
def match_keypoints(x_A, x_B, warp0, certainty0, warp1, certainty1):
    assert len(warp0.shape) == 3 and int(warp0.shape[2]) == 4 and len(warp1.shape) == 3 and int(warp1.shape[2]) == 4, str(warp0.shape)
    # warp B keypoints into image A
    x_A_from_B = F.grid_sample(warp1[:, :, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
    cert_A_from_B = F.grid_sample(certainty1[None, None, :, :], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
    # match in the coordinate system of A
    D = torch.cdist(x_A, x_A_from_B)
    inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
    return inds_A, inds_B, cert_A_from_B[inds_B]


def recover_pose(E, kpts0, kpts1, K0, K1, mask):
    best_num_inliers = 0
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0_n = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
    kpts1_n = (K1inv @ (kpts1-K1[None,:2,2]).T).T

    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t, mask.ravel() > 0)
    return ret



# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0 = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
    kpts1 = (K1inv @ (kpts1-K1[None,:2,2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret

def estimate_pose_uncalibrated(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    method = cv2.USAC_ACCURATE
    F, mask = cv2.findFundamentalMat(
        kpts0, kpts1, ransacReprojThreshold=norm_thresh, confidence=conf, method=method, maxIters=10000
    )
    E = K1.T@F@K0
    ret = None
    if E is not None:
        best_num_inliers = 0
        K0inv = np.linalg.inv(K0[:2,:2])
        K1inv = np.linalg.inv(K1[:2,:2])

        kpts0_n = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
        kpts1_n = (K1inv @ (kpts1-K1[None,:2,2]).T).T
 
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret

def unnormalize_coords(x_n,h,w):
    x = torch.stack(
        (w * (x_n[..., 0] + 1) / 2, h * (x_n[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    return x


def rotate_intrinsic(K, n):
    base_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot = np.linalg.matrix_power(base_rot, n)
    return rot @ K


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def compute_pose_error_T(T1, T2, R, t):
    R1 = T1[:3, :3]
    t1 = T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    R_est, t_est = compute_relative_pose(R1, t1, R2, t2)
    error_t = angle_error_vec(t_est.squeeze(), t)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R_est, R)
    return error_t, error_R

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    #accuracies = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
        #accuracy = np.mean(errors < t)  # Fraction of errors below the threshold
        #accuracies.append(accuracy)
    return aucs#, accuracies








def flow_to_pixel_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return flow

to_pixel_coords = flow_to_pixel_coords # just an alias

def flow_to_normalized_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                2 * (flow[..., 0]) / w1 - 1,
                2 * (flow[..., 1]) / h1 - 1,
            ),
            axis=-1,
        )
    )
    return flow

to_normalized_coords = flow_to_normalized_coords # just an alias

def warp_to_pixel_coords(warp, h1, w1, h2, w2):
    warp1 = warp[..., :2]
    warp1 = (
        torch.stack(
            (
                w1 * (warp1[..., 0] + 1) / 2,
                h1 * (warp1[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    warp2 = warp[..., 2:]
    warp2 = (
        torch.stack(
            (
                w2 * (warp2[..., 0] + 1) / 2,
                h2 * (warp2[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return torch.cat((warp1,warp2), dim=-1)



def signed_point_line_distance(point, line, eps: float = 1e-9):
    r"""Return the distance from points to lines.

    Args:
       point: (possibly homogeneous) points :math:`(*, N, 2 or 3)`.
       line: lines coefficients :math:`(a, b, c)` with shape :math:`(*, N, 3)`, where :math:`ax + by + c = 0`.
       eps: Small constant for safe sqrt.

    Returns:
        the computed distance with shape :math:`(*, N)`.
    """

    if not point.shape[-1] in (2, 3):
        raise ValueError(f"pts must be a (*, 2 or 3) tensor. Got {point.shape}")

    if not line.shape[-1] == 3:
        raise ValueError(f"lines must be a (*, 3) tensor. Got {line.shape}")

    numerator = (line[..., 0] * point[..., 0] + line[..., 1] * point[..., 1] + line[..., 2])
    denominator = line[..., :2].norm(dim=-1)

    return numerator / (denominator + eps)


def signed_left_to_right_epipolar_distance(pts1, pts2, Fm):
    r"""Return one-sided epipolar distance for correspondences given the fundamental matrix.

    This method measures the distance from points in the right images to the epilines
    of the corresponding points in the left images as they reflect in the right images.

    Args:
       pts1: correspondences from the left images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       pts2: correspondences from the right images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.

    Returns:
        the computed Symmetrical distance with shape :math:`(*, N)`.
    """
    import kornia
    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.shape[-1] == 2:
        pts1 = kornia.geometry.convert_points_to_homogeneous(pts1)

    F_t = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2 = pts1 @ F_t

    return signed_point_line_distance(pts2, line1_in_2)
