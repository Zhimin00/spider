import json
from spider.benchmarks import AerialMegaDepthPoseEstimationBenchmark, MegaDepthPoseEstimationBenchmark, ScanNetBenchmark
from spider.model import SPIDER_two
import torch
import numpy as np
import random
from argparse import ArgumentParser
from tqdm import tqdm
import os
import sys


def test_mega_two(model, name, coarse_size=512, fine_size=1600):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark('./data/megadepth/megadepth_test_1500')
    mega1500_results = mega1500_benchmark.benchmark_two(model, model_name=name, coarse_size=coarse_size, fine_size=fine_size)
    # json.dump(mega1500_results, open(f"./results/mega1500_two_{name}.json", "w"))

def test_scannet_two(model, name, coarse_size=512, fine_size=1600):
    scannet1500_benchmark = ScanNetBenchmark('./data/scannet1500')
    scan1500_results = scannet1500_benchmark.benchmark_two(model,  model_name=name, coarse_size=coarse_size, fine_size=fine_size)
    # json.dump(scan1500_results, open(f"./results/scan1500_two_{name}.json", "w"))

def test_aerial_two(model, name, coarse_size=512, fine_size=1600):
    aerial_benchmark = AerialMegaDepthPoseEstimationBenchmark('./data/megadepth_aerial_data/megadepth_aerial_processed')
    aerial_results = aerial_benchmark.benchmark_two(model, model_name=name, coarse_size=coarse_size, fine_size=fine_size)
    # json.dump(aerial_results, open(f"./results/aerial_two_{name}.json", "w"))


if __name__ == "__main__":
    device = "cuda"
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='spider_two', type=str)
    parser.add_argument("--coarse_size", default=512, type=int)
    parser.add_argument("--fine_size", default=512, type=int)
    args, _ = parser.parse_known_args()

    two_model = SPIDER_two.from_pretrained('./spider_two.pth').to(device)
 
    experiment_name = args.exp_name
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic
    torch.backends.cudnn.benchmark = False  
    test_scannet_two(two_model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)
    test_aerial_two(two_model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)
    test_mega_two(two_model, experiment_name, coarse_size=args.coarse_size, fine_size=args.fine_size)