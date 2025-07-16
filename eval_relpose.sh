CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --dataset aerial &
CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet &
CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega1500 &
CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --dataset hpatches &

wait

echo "All scripts have finished running."
