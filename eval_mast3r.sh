# CUDA_VISIBLE_DEVICES=1 python eval_mast3r.py --dataset aerial --model_name mast3r --exp_name mast3r &
# CUDA_VISIBLE_DEVICES=2 python eval_mast3r.py --dataset mega1500 --model_name mast3r --exp_name mast3r &
CUDA_VISIBLE_DEVICES=0 python eval_mast3r.py --dataset scannet --model_name mast3r --exp_name mast3r --fine_size 1296 &
# CUDA_VISIBLE_DEVICES=0 python eval_mast3r.py --dataset aerial --model_name mast3r --exp_name mast3r512 &
# CUDA_VISIBLE_DEVICES=2 python eval_mast3r.py --dataset mega1500 --model_name mast3r --exp_name mast3r512 &
# CUDA_VISIBLE_DEVICES=1 python eval_mast3r.py --dataset scannet --model_name mast3r --exp_name mast3r512 &
CUDA_VISIBLE_DEVICES=2 python eval_mast3r.py --dataset scannet --model_name aerial-mast3r --exp_name aerial-mast3r --fine_size 1296 &
# CUDA_VISIBLE_DEVICES=3 python eval_mast3r.py --dataset scannet --model_name aerial-mast3r --exp_name aerial-mast3r512 &
wait

echo "All scripts have finished running."

# CUDA_VISIBLE_DEVICES=1 python eval_mast3r.py --dataset aerial --model_name aerial-mast3r --exp_name aerial-mast3r &
# CUDA_VISIBLE_DEVICES=2 python eval_mast3r.py --dataset mega --model_name aerial-mast3r --exp_name aerial-mast3r &
# CUDA_VISIBLE_DEVICES=3 python eval_mast3r.py --dataset scannet --model_name aerial-mast3r --exp_name aerial-mast3r &
# wait

# echo "All scripts have finished running."