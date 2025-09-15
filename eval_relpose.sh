# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --dataset aerial --fine_size 512 --exp_name dad-spider-512 &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega1500 --fine_size 512 --exp_name dad-spider-512 &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet --fine_size 512 --exp_name dad-spider-512 &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet --fine_size 1600 --exp_name dad-spider-1600 &
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --dataset aerial --fine_size 1600 --exp_name dad-spider-1600 &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega1500 --fine_size 1600 --exp_name dad-spider-1600 &

CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset hpatches --fine_size 512 --exp_name spider-512 &
CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset hpatches --fine_size 1600 --exp_name spider-1600  &
wait

echo "All scripts have finished running."
