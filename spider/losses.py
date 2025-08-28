from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import math
import pdb

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.05):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x1_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest) 
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp
        
        
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]
    
    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0
    
def get_gt_warp(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):
    
    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ],
            indexing = 'ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.clone()
    K[:, 0, 2] += 0.5
    K[:, 1, 2] += 0.5
    return K

class PCK(nn.Module):
    def __init__(self, attenuate_cert=False):
        super().__init__()
        self.attenuate_cert = attenuate_cert
    
    def match(self, corresps):
        finest_scale = 1
        im_A_to_im_B = corresps[finest_scale]["flow"] 
        b, _, h, w = corresps[finest_scale]["certainty"].shape
        low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(h, w), align_corners=False, mode="bilinear"
                )
        cert_clamp = 0
        factor = 0.5
        low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)
        certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
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
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        return (
                warp,
                certainty[:, 0]
            )
        
    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches, is_2port, true_h2, true_w2):
        b, h1, w1, d = dense_matches.shape
        h2, w2 = depth2.shape[1:3]
        device = depth2.device
        true_w2 = true_w2.to(device)[:, None, None]
        true_h2 = true_h2.to(device)[:, None, None]
        with torch.no_grad():
            x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
            mask, x2 = warp_kpts(
                x1.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            x2 = torch.stack(
                (w2 * (x2[..., 0] + 1) / 2, h2 * (x2[..., 1] + 1) / 2), dim=-1
            )
            x2[is_2port] = x2[is_2port][..., [1,0]]
            prob = mask.float().reshape(b, h1, w1)
        x2_hat = dense_matches[..., 2:]
        x2_hat = torch.stack(
            (true_w2 * (x2_hat[..., 0] + 1) / 2, true_h2 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        epe = gd.float().mean()
        return epe, pck_1, pck_3, pck_5, prob

    def forward(self, view1, view2, corresps):
        img2 = view2['img']
        B, THREE, h2, w2 = img2.shape
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        height2, width2 = shape2.T
        is_2port = (width2 < height2)
        T1 = view1['camera_pose']
        T2 = view2['camera_pose']
        T_1to2 = torch.matmul(torch.linalg.inv(T2), T1)
        K1 = opencv_to_colmap_intrinsics(view1['camera_intrinsics'])
        K2 = opencv_to_colmap_intrinsics(view2['camera_intrinsics'])
        # K1 = view1['camera_intrinsics']
        # K2 = view2['camera_intrinsics']
        # pdb.set_trace()
        im_A_to_im_B = corresps[1]["flow"] 
        _, _, h, w = im_A_to_im_B.shape
        gt_warp, gt_prob = get_gt_warp(                
                view1['depthmap'],
                view2['depthmap'],
                T_1to2,
                K1,
                K2,
                H=h,
                W=w,
            )
        x2 = gt_warp.float()
        prob = gt_prob
        
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
        im_A_to_im_B[is_2port] = im_A_to_im_B[is_2port][..., [1,0]]

        x2_hat = torch.clamp(im_A_to_im_B, -1, 1)
        
        x2 = torch.stack(
                (w2 * (x2[..., 0] + 1) / 2, h2 * (x2[..., 1] + 1) / 2), dim=-1
            )
        x2_hat = torch.stack(
                (w2 * (x2_hat[..., 0] + 1) / 2, h2 * (x2_hat[..., 1] + 1) / 2), dim=-1
            )  

        gd = (x2_hat - x2).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        epe = gd.float().mean()  
        # pdb.set_trace()
        # matches, certainty = self.match(corresps)

        # epe, pck_1, pck_3, pck_5, prob = self.geometric_dist(
        #      view1['depthmap'], view2['depthmap'], T_1to2, K1, K2, matches, is_2port, height2, width2
        # )
       
        return epe, dict(epe=epe, pck1 = pck_1, pck3 = pck_3, pck5 = pck_5)

   

class RobustLosses(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices ##ground coordinates in res scale [B,H,W,2]
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        return losses
    
    
    def forward(self, view1, view2, corresps):
        img2 = view2['img']
        B = img2.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        height2, width2 = shape2.T
        is_2port = (width2 < height2)

        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        details = {}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),
                )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                # _ = 1
                b, _, h, w = scale_certainty.shape
            T1 = view1['camera_pose']
            T2 = view2['camera_pose']
            T_1to2 = torch.matmul(torch.linalg.inv(T2), T1)
            K1 = opencv_to_colmap_intrinsics(view1['camera_intrinsics'])
            K2 = opencv_to_colmap_intrinsics(view2['camera_intrinsics'])
            # pdb.set_trace()
            gt_warp, gt_prob = get_gt_warp(                
                view1['depthmap'],
                view2['depthmap'],
                T_1to2,
                K1,
                K2,
                H=h,
                W=w,
            )
            gt_warp[is_2port] = gt_warp[is_2port][..., [1,0]]


            x2 = gt_warp.float()
            prob = gt_prob
            
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
                details.update(gm_cls_losses)
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
                details.update(gm_flow_losses)
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
                details.update(delta_cls_losses)
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
                details.update(delta_regression_losses)
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()

        return tot_loss, dict(warp_loss = float(tot_loss), **details)
    
class RobustLosses_fm(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices ##ground coordinates in res scale [B,H,W,2]
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        return losses
    
    
    def forward(self, view1, view2, corresps):
        img2 = view2['img']
        B = img2.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        height2, width2 = shape2.T
        is_2port = (width2 < height2)

        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        details = {}
        scale = 1
        scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
            corresps["certainty"],
            corresps.get("flow_pre_delta"),
            corresps.get("delta_cls"),
            corresps.get("offset_scale"),
            corresps.get("gm_cls"),
            corresps.get("gm_certainty"),
            corresps["flow"],
            corresps.get("gm_flow"),
            )
        if flow_pre_delta is not None:
            flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
            b, h, w, d = flow_pre_delta.shape
        else:
            # _ = 1
            b, _, h, w = scale_certainty.shape
        T1 = view1['camera_pose']
        T2 = view2['camera_pose']
        T_1to2 = torch.matmul(torch.linalg.inv(T2), T1)
        K1 = opencv_to_colmap_intrinsics(view1['camera_intrinsics'])
        K2 = opencv_to_colmap_intrinsics(view2['camera_intrinsics'])
        # pdb.set_trace()
        gt_warp, gt_prob = get_gt_warp(                
            view1['depthmap'],
            view2['depthmap'],
            T_1to2,
            K1,
            K2,
            H=h,
            W=w,
        )
        gt_warp[is_2port] = gt_warp[is_2port][..., [1,0]]


        x2 = gt_warp.float()
        prob = gt_prob

        if scale_gm_cls is not None:
            gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
            gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
            tot_loss = tot_loss + scale_weights[scale] * gm_loss
            details.update(gm_cls_losses)
        elif scale_gm_flow is not None:
            gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
            gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
            tot_loss = tot_loss + scale_weights[scale] * gm_loss
            details.update(gm_flow_losses)
        
        if delta_cls is not None:
            delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
            delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
            tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            details.update(delta_cls_losses)
        else:
            delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
            reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
            tot_loss = tot_loss + scale_weights[scale] * reg_loss
            details.update(delta_regression_losses)

        return tot_loss, dict(warp_loss = float(tot_loss), **details)