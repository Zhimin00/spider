import torch
import numpy as np
import tqdm
from romatch.datasets import MegadepthBuilder, Aerial_MegaDepth
from romatch.utils import warp_kpts, depthmap_to_absolute_camera_coordinates
from torch.utils.data import ConcatDataset
import romatch
import pdb
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.utils.geometry import geotrf
from dust3r.utils.device import collate_with_cat

class MegadepthDenseBenchmark:
    def __init__(self, data_root="data/megadepth", h = 384, w = 512, num_samples = 2000) -> None:
        mega = MegadepthBuilder(data_root=data_root)
        self.dataset = ConcatDataset(
            mega.build_scenes(split="test_loftr", ht=h, wt=w)
        )  # fixed resolution of 384,512
        self.num_samples = num_samples

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches):
        b, h1, w1, d = dense_matches.shape
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
                (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = mask.float().reshape(b, h1, w1)
        x2_hat = dense_matches[..., 2:]
        x2_hat = torch.stack(
            (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        return gd, pck_1, pck_3, pck_5, prob

    def benchmark(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=self.num_samples
            )
            B = batch_size
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=B, num_workers=batch_size, sampler=sampler
            )
            for idx, data in tqdm.tqdm(enumerate(dataloader), disable = romatch.RANK > 0):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
                    data["im_A"].cuda(),
                    data["im_B"].cuda(),
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                )
                matches, certainty = model.match(im_A, im_B, batched=True)
                gd, pck_1, pck_3, pck_5, prob = self.geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )
                if romatch.DEBUG_MODE:
                    from romatch.utils.utils import tensor_to_pil
                    import torch.nn.functional as F
                    path = "vis"
                    H, W = model.get_output_resolution()
                    white_im = torch.ones((B,1,H,W),device="cuda")
                    im_B_transfer_rgb = F.grid_sample(
                        im_B.cuda(), matches[:,:,:W, 2:], mode="bilinear", align_corners=False
                    )
                    warp_im = im_B_transfer_rgb
                    c_b = certainty[:,None]#(certainty*0.9 + 0.1*torch.ones_like(certainty))[:,None]
                    vis_im = c_b * warp_im + (1 - c_b) * white_im
                    for b in range(B):
                        import os
                        os.makedirs(f"{path}/{model.name}/{idx}_{b}_{H}_{W}",exist_ok=True)
                        tensor_to_pil(vis_im[b], unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/warp.jpg")
                        tensor_to_pil(im_A[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_A.jpg")
                        tensor_to_pil(im_B[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_B.jpg")


                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot = (
                    gd_tot + gd.mean(),
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )
        return {
            "epe": gd_tot.item() / len(dataloader),
            "mega_pck_1": pck_1_tot.item() / len(dataloader),
            "mega_pck_3": pck_3_tot.item() / len(dataloader),
            "mega_pck_5": pck_5_tot.item() / len(dataloader),
        }

    def benchmark_adapter(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=self.num_samples
            )
            B = batch_size
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=B, num_workers=batch_size, sampler=sampler
            )
            for idx, data in tqdm.tqdm(enumerate(dataloader), disable = romatch.RANK > 0):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2, domainid = (
                    data["im_A"].cuda(),
                    data["im_B"].cuda(),
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                    data['domainid'].cuda(),
                )
                matches, certainty = model.match(im_A, im_B, domainid, batched=True)
                gd, pck_1, pck_3, pck_5, prob = self.geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )
                if romatch.DEBUG_MODE:
                    from romatch.utils.utils import tensor_to_pil
                    import torch.nn.functional as F
                    path = "vis"
                    H, W = model.get_output_resolution()
                    white_im = torch.ones((B,1,H,W),device="cuda")
                    im_B_transfer_rgb = F.grid_sample(
                        im_B.cuda(), matches[:,:,:W, 2:], mode="bilinear", align_corners=False
                    )
                    warp_im = im_B_transfer_rgb
                    c_b = certainty[:,None]#(certainty*0.9 + 0.1*torch.ones_like(certainty))[:,None]
                    vis_im = c_b * warp_im + (1 - c_b) * white_im
                    for b in range(B):
                        import os
                        os.makedirs(f"{path}/{model.name}/{idx}_{b}_{H}_{W}",exist_ok=True)
                        tensor_to_pil(vis_im[b], unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/warp.jpg")
                        tensor_to_pil(im_A[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_A.jpg")
                        tensor_to_pil(im_B[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_B.jpg")


                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot = (
                    gd_tot + gd.mean(),
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )
        return {
            "epe": gd_tot.item() / len(dataloader),
            "mega_pck_1": pck_1_tot.item() / len(dataloader),
            "mega_pck_3": pck_3_tot.item() / len(dataloader),
            "mega_pck_5": pck_5_tot.item() / len(dataloader),
        }

class AerialMegadepthDenseBenchmark:
    def __init__(self, data_root='/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth/megadepth_aerial_processed', h = 384, w = 512, num_samples = 2000) -> None:
        self.dataset = Aerial_MegaDepth(data_root, 'val', ht=h,wt=w,) # fixed resolution of 384,512
        self.num_samples = num_samples

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches):
        b, h1, w1, d = dense_matches.shape
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
                (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = mask.float().reshape(b, h1, w1)
        x2_hat = dense_matches[..., 2:]
        x2_hat = torch.stack(
            (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        return gd, pck_1, pck_3, pck_5, prob

    def benchmark(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=self.num_samples
            )
            B = batch_size
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=B, num_workers=batch_size, sampler=sampler
            )
            for idx, data in tqdm.tqdm(enumerate(dataloader), disable = romatch.RANK > 0):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
                    data["im_A"].cuda(),
                    data["im_B"].cuda(),
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                )
                matches, certainty = model.match(im_A, im_B, batched=True)
                gd, pck_1, pck_3, pck_5, prob = self.geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )
                if romatch.DEBUG_MODE:
                    from romatch.utils.utils import tensor_to_pil
                    import torch.nn.functional as F
                    path = "vis"
                    H, W = model.get_output_resolution()
                    white_im = torch.ones((B,1,H,W),device="cuda")
                    im_B_transfer_rgb = F.grid_sample(
                        im_B.cuda(), matches[:,:,:W, 2:], mode="bilinear", align_corners=False
                    )
                    warp_im = im_B_transfer_rgb
                    c_b = certainty[:,None]#(certainty*0.9 + 0.1*torch.ones_like(certainty))[:,None]
                    vis_im = c_b * warp_im + (1 - c_b) * white_im
                    for b in range(B):
                        import os
                        os.makedirs(f"{path}/{model.name}/{idx}_{b}_{H}_{W}",exist_ok=True)
                        tensor_to_pil(vis_im[b], unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/warp.jpg")
                        tensor_to_pil(im_A[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_A.jpg")
                        tensor_to_pil(im_B[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_B.jpg")


                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot = (
                    gd_tot + gd.mean(),
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )
        return {
            "epe": gd_tot.item() / len(dataloader),
            "mega_pck_1": pck_1_tot.item() / len(dataloader),
            "mega_pck_3": pck_3_tot.item() / len(dataloader),
            "mega_pck_5": pck_5_tot.item() / len(dataloader),
        }
    
class MegadepthDenseBenchmark_depth:
    def __init__(self, data_root="data/megadepth", h = 384, w = 512, num_samples = 2000) -> None:
        mega = MegadepthBuilder(data_root=data_root)
        self.dataset = ConcatDataset(
            mega.build_scenes(split="test_loftr", ht=h, wt=w)
        )  # fixed resolution of 384,512
        self.num_samples = num_samples

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches):
        b, h1, w1, d = dense_matches.shape
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
                (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = mask.float().reshape(b, h1, w1)
        x2_hat = dense_matches[..., 2:]
        x2_hat = torch.stack(
            (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        return gd, pck_1, pck_3, pck_5, prob
    def depth_dist(self, depth1, depth2, K1, K2, T1, T2, pts1, pts_conf1, pts2, pts_conf2):
        gt_pts1, gt_valid1 = depthmap_to_absolute_camera_coordinates(depth1, K1, T1)
        gt_pts2, gt_valid2 = depthmap_to_absolute_camera_coordinates(depth2, K2, T2)
        with torch.no_grad():
            in_camera1 = inv(T1)
            gt_pts1 = geotrf(in_camera1, gt_pts1)  # B,H,W,3
            gt_pts2 = geotrf(in_camera1, gt_pts2)  # B,H,W,3
            valid1 = gt_valid1.clone() # B,H,W
            valid2 = gt_valid2.clone()
            pr_pts1, pr_pts2 = normalize_pointcloud(pts1, pts2, 'avg_dis', valid1, valid2)
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, 'avg_dis', valid1, valid2)
            gd1 = (pr_pts1 - gt_pts1).norm(dim=-1)
            gd1 = gd1[valid1 == 1]
            gd2 = (pr_pts2 - gt_pts2).norm(dim=-1)
            gd2 = gd2[valid2 == 1]

            pck1_1 = (gd1 < 1.0).float().mean()
            pck1_3 = (gd1 < 3.0).float().mean()
            pck1_5 = (gd1 < 5.0).float().mean()
            pck2_1 = (gd2 < 1.0).float().mean()
            pck2_3 = (gd2 < 3.0).float().mean()
            pck2_5 = (gd2 < 5.0).float().mean()
            return gd1, pck1_1, pck1_3, pck1_5, valid1, gd2, pck2_1, pck2_3, pck2_5, valid2

    def benchmark(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            gd1_tot, pck1_1_tot, pck1_3_tot, pck1_5_tot = 0.0, 0.0, 0.0, 0.0
            gd2_tot, pck2_1_tot, pck2_3_tot, pck2_5_tot = 0.0, 0.0, 0.0, 0.0
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=self.num_samples
            )
            B = batch_size
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=B, num_workers=batch_size, sampler=sampler
            )
            for idx, data in tqdm.tqdm(enumerate(dataloader), disable = romatch.RANK > 0):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2, T1, T2 = (
                    data["im_A"].cuda(),
                    data["im_B"].cuda(),
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                    data["T1"].cuda(),
                    data["T2"].cuda(),
                )
                matches, certainty, pts1, pts_conf1, pts2, pts_conf2 = model.match(im_A, im_B, batched=True)
                gd, pck_1, pck_3, pck_5, prob = self.geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )
                gd1, pck1_1, pck1_3, pck1_5, valid1, gd2, pck2_1, pck2_3, pck2_5, valid2 = self.depth_dist(depth1, depth2, K1, K2, T1, T2, pts1, pts_conf1, pts2, pts_conf2)
                
                if romatch.DEBUG_MODE:
                    from romatch.utils.utils import tensor_to_pil
                    import torch.nn.functional as F
                    path = "vis"
                    H, W = model.get_output_resolution()
                    white_im = torch.ones((B,1,H,W),device="cuda")
                    im_B_transfer_rgb = F.grid_sample(
                        im_B.cuda(), matches[:,:,:W, 2:], mode="bilinear", align_corners=False
                    )
                    warp_im = im_B_transfer_rgb
                    c_b = certainty[:,None]#(certainty*0.9 + 0.1*torch.ones_like(certainty))[:,None]
                    vis_im = c_b * warp_im + (1 - c_b) * white_im
                    for b in range(B):
                        import os
                        os.makedirs(f"{path}/{model.name}/{idx}_{b}_{H}_{W}",exist_ok=True)
                        tensor_to_pil(vis_im[b], unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/warp.jpg")
                        tensor_to_pil(im_A[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_A.jpg")
                        tensor_to_pil(im_B[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_B.jpg")


                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot = (
                    gd_tot + gd.mean(),
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )
                gd1_tot, pck1_1_tot, pck1_3_tot, pck1_5_tot = (
                    gd1_tot + gd1.mean(),
                    pck1_1_tot + pck1_1,
                    pck1_3_tot + pck1_3,
                    pck1_5_tot + pck1_5,
                )
                gd2_tot, pck2_1_tot, pck2_3_tot, pck2_5_tot = (
                    gd2_tot + gd2.mean(),
                    pck2_1_tot + pck2_1,
                    pck2_3_tot + pck2_3,
                    pck2_5_tot + pck2_5,
                )
        return {
            "epe": gd_tot.item() / len(dataloader),
            "mega_pck_1": pck_1_tot.item() / len(dataloader),
            "mega_pck_3": pck_3_tot.item() / len(dataloader),
            "mega_pck_5": pck_5_tot.item() / len(dataloader),
            "pts1_epe": gd1_tot.item() / len(dataloader),
            "pts1_pck_1": pck1_1_tot.item() / len(dataloader),
            "pts1_pck_3": pck1_3_tot.item() / len(dataloader),
            "pts1_pck_5": pck1_5_tot.item() / len(dataloader),
            "pts2_epe": gd2_tot.item() / len(dataloader),
            "pts2_pck_1": pck2_1_tot.item() / len(dataloader),
            "pts2_pck_3": pck2_3_tot.item() / len(dataloader),
            "pts2_pck_5": pck2_5_tot.item() / len(dataloader),
        }
