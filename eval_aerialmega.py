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


def test_aerial(model, name, coarse_size=512, fine_size=1600):
    aerial_benchmark = AerialMegaDepthPoseEstimationBenchmark('/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth_aerial_processed')
    aerial_results = aerial_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(aerial_results, open(f"./results/aerial_{name}.json", "w"))

def test_mega1500(model, name, coarse_size=512, fine_size=1600):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark('/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500')
    mega1500_results = mega1500_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(mega1500_results, open(f"./results/mega1500_{name}.json", "w"))

if __name__ == "__main__":
    device = "cuda"
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='spider', type=str)
    parser.add_argument("--dataset", default='aerial', type=str)
    parser.add_argument("--coarse_size", default=512, type=int)
    parser.add_argument("--fine_size", default=1600, type=int)
    args, _ = parser.parse_known_args()

    model = SPIDER.from_pretrained("./model_weights/spider.pth").to(device)
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
    
    if args.dataset == 'mega1500':
        test_mega1500(model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)
    elif args.dataset == 'aerial':
        test_aerial(model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)