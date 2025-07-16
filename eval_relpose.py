import json
from spider.benchmarks import AerialMegaDepthPoseEstimationBenchmark, MegaDepthPoseEstimationBenchmark, ScanNetBenchmark, HpatchesHomogBenchmark
from spider.model import SPIDER
import torch
import numpy as np
import random
from argparse import ArgumentParser

def test_aerial(model, name):
    aerial_benchmark = AerialMegaDepthPoseEstimationBenchmark('/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth_aerial_processed')
    aerial_results = aerial_benchmark.benchmark(model)
    json.dump(aerial_results, open(f"./results/aerial_{name}.json", "w"))

def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark('/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500')
    mega1500_results = mega1500_benchmark.benchmark(model)
    json.dump(mega1500_results, open(f"./results/mega1500_{name}.json", "w"))


def test_scannet1500(model, name):
    scannet1500_benchmark = ScanNetBenchmark('/cis/net/r24a/data/zshao/data/scannet1500')
    scan1500_results = scannet1500_benchmark.benchmark(model)
    json.dump(scan1500_results, open(f"./results/scan1500_{name}.json", "w"))

def test_hpatches(model, name):
    hpatches_benchmark = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release')
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"./results/hpatches_{name}.json", "w"))
    
    hpatches_benchmark_view = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release', seqs_dir='hpatches-sequences-v')
    hpatches_view_results = hpatches_benchmark_view.benchmark(model)
    json.dump(hpatches_view_results, open(f"./results/hpatches_view_{name}.json", "w"))

    hpatches_benchmark_illu = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release', seqs_dir='hpatches-sequences-i')
    hpatches_illu_results = hpatches_benchmark_illu.benchmark(model)
    json.dump(hpatches_illu_results, open(f"./results/hpatches_illumination_{name}.json", "w"))


if __name__ == "__main__":
    device = "cuda"
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='spider', type=str)
    parser.add_argument("--dataset", default='aerial', type=str)
    args, _ = parser.parse_known_args()

    model = SPIDER.from_pretrained("/cis/home/zshao14/checkpoints/spider_warp/checkpoint-best.pth").to(device)
    
    experiment_name = "spider-best_1344"
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic
    torch.backends.cudnn.benchmark = False  
    if args.dataset == 'scannet':
        test_scannet1500(model, experiment_name) 
    elif args.dataset == 'mega1500':
        test_mega1500(model, experiment_name)
    elif args.dataset == 'aerial':
        test_aerial(model, experiment_name)
    elif args.dataset =='hpatches':
        test_hpatches(model, experiment_name)