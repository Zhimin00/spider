CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --dataset aerial --fine_size 1600 --exp_name spider-0730best &
CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega1500 --fine_size 1600 --exp_name spider-0730best &
CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet --fine_size 512 --exp_name spider-0730best &
# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --dataset hpatches --fine_size 1600 --exp_name spider-0730best &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset aerial --fine_size 1600 --exp_name spider-0727best_1600 &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega1500 --fine_size 1600 --exp_name spider-0727best_1600 &

# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --dataset hpatches &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet &
wait

echo "All scripts have finished running."
