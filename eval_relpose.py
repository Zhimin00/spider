import json
from spider.benchmarks import AerialMegaDepthPoseEstimationBenchmark, MegaDepthPoseEstimationBenchmark, ScanNetBenchmark, HpatchesHomogBenchmark, spider_match_path, dad_spider_match_path
from spider.model import SPIDER
import torch
import numpy as np
import random
from argparse import ArgumentParser
from mast3r.model import AsymmetricMASt3R
from spider.utils.utils import compute_relative_pose, estimate_pose, compute_pose_error, pose_auc
from tqdm import tqdm
import os
import sys
import pdb
dad_path = os.path.abspath('/cis/home/zshao14/Downloads/dad')
sys.path.insert(0, dad_path)
import dad as dad_detector


def test_aerial(model, name, coarse_size=512, fine_size=1344):
    aerial_benchmark = AerialMegaDepthPoseEstimationBenchmark('/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth_aerial_processed')
    aerial_results = aerial_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(aerial_results, open(f"./results/aerial_{name}.json", "w"))

def test_aerial_2dand3d(model1, model2, name, coarse_size=512, fine_size=1344):
    aerial_benchmark = AerialMegaDepthPoseEstimationBenchmark('/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth_aerial_processed')
    aerial_results = aerial_benchmark.benchmark_2dand3d(model1, model2, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(aerial_results, open(f"./results/aerial_2dand3d_{name}.json", "w"))

def test_mega1500(model, name, coarse_size=512, fine_size=1344):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark('/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500')
    mega1500_results = mega1500_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(mega1500_results, open(f"./results/mega1500_{name}.json", "w"))

def test_mega_fine(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark('/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500')
    mega1500_results = mega1500_benchmark.benchmark_coarse_to_fine(model)
    json.dump(mega1500_results, open(f"./results/mega1500_coarse_to_fine_{name}.json", "w"))

def test_mega_2dand3d(model1, model2, name, coarse_size=512, fine_size=1344):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark('/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500')
    mega1500_results = mega1500_benchmark.benchmark_2dand3d(model1, model2, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(mega1500_results, open(f"./results/mega1500_2dand3d_{name}.json", "w"))


def test_scannet1500(model, name, coarse_size=512, fine_size=1344):
    scannet1500_benchmark = ScanNetBenchmark('/cis/net/r24a/data/zshao/data/scannet1500')
    scan1500_results = scannet1500_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(scan1500_results, open(f"./results/scan1500_{name}.json", "w"))


def test_hpatches(model, name, coarse_size=512, fine_size=1344):
    hpatches_benchmark = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release')
    hpatches_results = hpatches_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(hpatches_results, open(f"./results/hpatches_{name}.json", "w"))
    
    # hpatches_benchmark_view = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release', seqs_dir='hpatches-sequences-v')
    # hpatches_view_results = hpatches_benchmark_view.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    # json.dump(hpatches_view_results, open(f"./results/hpatches_view_{name}.json", "w"))

    # hpatches_benchmark_illu = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release', seqs_dir='hpatches-sequences-i')
    # hpatches_illu_results = hpatches_benchmark_illu.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    # json.dump(hpatches_illu_results, open(f"./results/hpatches_illumination_{name}.json", "w"))

def test_jhu(model, name, coarse_size=512, fine_size=1600):
    # detector = dad_detector.load_DaD()

    data_root = '/cis/net/io96/data/zshao/JHU-ULTRA-360/pairs'
    scene_names = ["24_Clark.npy",
                # "10_AMES.npy",
                # "24_Clark.npy",
                # "28_Garland.npy",
                # "30_Gilman.npy",
                # "34_Hackerman.npy",
                # "35_Hodson.npy",
                # "48_Latrobe.npy",
                # "49_Levering.npy",
                # "53_Maryland.npy",
                # "54_Mason.npy",
                # # "77_Shaffer.npy",
                # "78_Shriver.npy",
            ]
            
    scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in scene_names
        ]
    results = {}
    thresholds = [5, 10, 20, 30]
    tot_e_t, tot_e_R, tot_e_pose = [], [], []
    for scene_ind in range(len(scenes)):
        scene_e_pose = []
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

            kpts1, kpts2, mconf, K1, K2 =  spider_match_path(im_A_path, im_B_path, K1_ori, K2_ori, model, device, coarse_size=coarse_size, fine_size=fine_size)
            kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
            # for _ in range(5):
                # shuffling = np.random.permutation(np.arange(len(kpts1)))
                # kpts1 = kpts1[shuffling]
                # kpts2 = kpts2[shuffling]
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
            scene_e_pose.append(e_pose)
        scene_e_pose = np.array(scene_e_pose)
        scene_auc = pose_auc(scene_e_pose, thresholds)
        print(f"{scene_name} auc: {scene_auc}")
        results.update({scene_name: scene_auc})
    tot_e_pose = np.array(tot_e_pose)
    
    auc = pose_auc(tot_e_pose, thresholds)
    acc_5 = (tot_e_pose < 5).mean()
    acc_10 = (tot_e_pose < 10).mean()
    acc_15 = (tot_e_pose < 15).mean()
    acc_20 = (tot_e_pose < 20).mean()
    acc_30 = (tot_e_pose < 30).mean()
    map_5 = acc_5
    map_10 = np.mean([acc_5, acc_10])
    map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
    results.update({ "auc_30": auc[3],
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            })
    print(auc)
    json.dump(results, open(f"./results/jhu_{name}.json", "w"))
if __name__ == "__main__":
    device = "cuda"
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='spider', type=str)
    parser.add_argument("--dataset", default='aerial', type=str)
    parser.add_argument("--coarse_size", default=512, type=int)
    parser.add_argument("--fine_size", default=1344, type=int)
    args, _ = parser.parse_known_args()

    model = SPIDER.from_pretrained("/cis/home/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth").to(device)
    # mast3r_model = AsymmetricMASt3R.from_pretrained("/cis/home/zshao14/checkpoints/checkpoint-aerial-mast3r.pth").to(device)
    # experiment_name = "spider-0727best_1024"
    experiment_name = args.exp_name
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic
    torch.backends.cudnn.benchmark = False  
    if args.dataset == 'scannet':
        test_scannet1500(model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size) 
    elif args.dataset == 'mega1500':
        test_mega1500(model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)
    elif args.dataset == 'aerial':
        test_aerial(model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)
    elif args.dataset =='hpatches':
        test_hpatches(model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)
    elif args.dataset =='mega_fine':
        test_mega_fine(model, experiment_name)
    elif args.dataset == 'jhu':
        test_jhu(model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)
    # elif args.dataset == 'mega_2dand3d':
        # test_mega_2dand3d(model, mast3r_model, experiment_name)
    # elif args.dataset == 'aerial_2dand3d':
        # test_aerial_2dand3d(model, mast3r_model, experiment_name)