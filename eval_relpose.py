import json
from spider.benchmarks import AerialMegaDepthPoseEstimationBenchmark, MegaDepthPoseEstimationBenchmark, ScanNetBenchmark, HpatchesHomogBenchmark
from spider.model import SPIDER
import torch
import numpy as np
import random
from argparse import ArgumentParser

def test_aerial(model, name, coarse_size=512, fine_size=1344):
    aerial_benchmark = AerialMegaDepthPoseEstimationBenchmark('/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth_aerial_processed')
    aerial_results = aerial_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(aerial_results, open(f"./results/aerial_{name}.json", "w"))

def test_mega1500(model, name, coarse_size=512, fine_size=1344):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark('/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500')
    mega1500_results = mega1500_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(mega1500_results, open(f"./results/mega1500_{name}.json", "w"))

def test_mega_fine(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark('/cis/net/r24a/data/zshao/data/megadepth/megadepth_test_1500')
    mega1500_results = mega1500_benchmark.benchmark_coarse_to_fine(model)
    json.dump(mega1500_results, open(f"./results/mega1500_coarse_to_fine_{name}.json", "w"))

def test_scannet1500(model, name, coarse_size=512, fine_size=1344):
    scannet1500_benchmark = ScanNetBenchmark('/cis/net/r24a/data/zshao/data/scannet1500')
    scan1500_results = scannet1500_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(scan1500_results, open(f"./results/scan1500_{name}.json", "w"))

def test_hpatches(model, name, coarse_size=512, fine_size=1344):
    hpatches_benchmark = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release')
    hpatches_results = hpatches_benchmark.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(hpatches_results, open(f"./results/hpatches_{name}.json", "w"))
    
    hpatches_benchmark_view = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release', seqs_dir='hpatches-sequences-v')
    hpatches_view_results = hpatches_benchmark_view.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(hpatches_view_results, open(f"./results/hpatches_view_{name}.json", "w"))

    hpatches_benchmark_illu = HpatchesHomogBenchmark('/cis/net/r24a/data/zshao/data/hpatches-sequence-release', seqs_dir='hpatches-sequences-i')
    hpatches_illu_results = hpatches_benchmark_illu.benchmark(model, coarse_size=coarse_size, fine_size=fine_size)
    json.dump(hpatches_illu_results, open(f"./results/hpatches_illumination_{name}.json", "w"))


if __name__ == "__main__":
    device = "cuda"
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='spider', type=str)
    parser.add_argument("--dataset", default='aerial', type=str)
    parser.add_argument("--coarse_size", default=512, type=int)
    parser.add_argument("--fine_size", default=1344, type=int)
    args, _ = parser.parse_known_args()

    model = SPIDER.from_pretrained("/cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth").to(device)
    
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