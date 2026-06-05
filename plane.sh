CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --dataset planary --exp_name spider-twoheads-concat-planary-1600-all &
CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset planary --exp_name spider-warp-planary-1600-all &
CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset planary --exp_name spider-msfr-planary-1600-all &
CUDA_VISIBLE_DEVICES=3 python eval_mast3r.py --dataset planary -model_name mast3r --exp_name mast3r-planary-1600-all&
CUDA_VISIBLE_DEVICES=4 python eval_mast3r.py --dataset planary --model_name aerial-mast3r --exp_name aerial-mast3r-planary-1600-all &
wait

echo "All scripts have finished running."
