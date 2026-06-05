# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --dataset aerial --fine_size 512 --exp_name dad-spider-512 &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega1500 --fine_size 512 --exp_name dad-spider-512 &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet --fine_size 512 --exp_name dad-spider-512 &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet --fine_size 1600 --exp_name dad-spider-1600 &
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --dataset aerial --fine_size 1600 --exp_name dad-spider-1600 &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega1500 --fine_size 1600 --exp_name dad-spider-1600 &
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --dataset scannet_concat --fine_size 512 --exp_name mast3r-spider-sample-512 &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega_concat --fine_size 512 --exp_name mast3r-spider-sample-512 &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset aerial_concat --fine_size 512 --exp_name mast3r-spider-sample-512 &
# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --dataset scannet_concat --fine_size 1600 --exp_name mast3r-spider-sample-1600 &
# wait
# CUDA_VISIBLE_DEVICES=6 python eval_relpose.py --dataset mega_concat --fine_size 1600 --exp_name mast3r-spider-sample-1600 &
# CUDA_VISIBLE_DEVICES=7 python eval_relpose.py --dataset aerial_concat --fine_size 1600 --exp_name mast3r-spider-sample-1600 &
# wait
# echo "All scripts have finished running."

# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet --fine_size 512 --exp_name spider-msfr-512 & 
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset aerial --fine_size 512 --exp_name spider-msfr-512 & 
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --dataset mega1500 --fine_size 512 --exp_name spider-msfr-512 & 
# wait
# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --dataset aerial --fine_size 512 --exp_name spider-msfr-512 & 
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --dataset scannet --fine_size 1600 --exp_name spider-msfr-1600 &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --dataset mega1500 --fine_size 1600 --exp_name spider-msfr-1600 & 
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --dataset aerial --fine_size 1600 --exp_name spider-msfr-1600 & 
# wait
# echo "All scripts have finished running."

# CUDA_VISIBLE_DEVICES=4 python eval_relpose.py --exp_name spider-msfr-512 --fine_size 512 &
# CUDA_VISIBLE_DEVICES=5 python eval_relpose.py --exp_name spider-msfr-1600 --fine_size 1600 &
# wait

# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --exp_name msfr-warp-concat-512 --fine_size 512 --dataset mega_concat &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --exp_name msfr-warp-concat-512 --fine_size 512 --dataset scannet_concat &
# wait 

# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --exp_name msfr-warp-concat-512 --fine_size 512 --dataset aerial_concat &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --exp_name msfr-warp-concat-1600-ab --fine_size 1600 --dataset mega_concat &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --exp_name msfr-warp-concat-1600-ab --fine_size 1600 --dataset aerial_concat &
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --exp_name msfr-warp-concat-1600-ab --fine_size 1600 --dataset scannet_concat &
# wait
# CUDA_VISIBLE_DEVICES=6 python eval_relpose.py --exp_name msfr-warp-concat-512-0.2 --fine_size 512 --dataset aerial_concat &
# CUDA_VISIBLE_DEVICES=7 python eval_relpose.py --exp_name msfr-warp-concat-1600-0.2 --fine_size 1600 --dataset aerial_concat &
#CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --exp_name spider_full_512 --dataset scannet --fine_size 512 &
#CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --exp_name spider_full_512 --dataset mega1500 --fine_size 512 &
#wait

# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --exp_name spider_full_512 --dataset aerial --fine_size 512 &
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --exp_name spider_full --dataset aerial &
#CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --exp_name spider_full --dataset mega1500 &
#CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --exp_name spider_full --dataset scannet &
# # CUDA_VISIBLE_DEVICES=3 python eval_mast3r.py --exp_name aerial-mast3r --model_name aerial-mast3r --dataset m07 &
# CUDA_VISIBLE_DEVICES=0 python eval_mast3r.py --dataset jhu_gps --model_name aerial-mast3r --exp_name aerial-mast3r-gg &
# CUDA_VISIBLE_DEVICES=0 python eval_mast3r.py --dataset jhu_gps --model_name mast3r --exp_name mast3r-gg &
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --exp_name spider-concat-ab-gg --dataset jhu_gps &
# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --exp_name spider-warp-gg --dataset jhu_gps &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --exp_name spider-msfr-gg2 --dataset jhu_gps &
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --exp_name spider-concat-ab512 --dataset m07 &

# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --exp_name spider-warp512 --dataset m07 &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --exp_name spider-msfr512 --dataset m07 &

# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --exp_name spider-warp-gg --dataset m07 &
# CUDA_VISIBLE_DEVICES=4 python eval_relpose.py --exp_name spider-msfr-gg --dataset m07 &
# CUDA_VISIBLE_DEVICES=5 python eval_mast3r.py --exp_name aerial-mast3r512-gg --model_name aerial-mast3r --fine_size 512 --dataset m07 &
# CUDA_VISIBLE_DEVICES=6 python eval_mast3r.py --exp_name mast3r512-gg --model_name mast3r --fine_size 512 --dataset m07 &

# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py --exp_name spider-warp-M07gg --dataset m07 &
# # CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --exp_name spider-msfr-M07gg --dataset m07 &
# CUDA_VISIBLE_DEVICES=1 python eval_mast3r.py --exp_name aerial-mast3r-M07gg --model_name aerial-mast3r --dataset m07 &
# CUDA_VISIBLE_DEVICES=3 python eval_mast3r.py --exp_name mast3r-M07gg --model_name mast3r --dataset m07 &
# CUDA_VISIBLE_DEVICES=2 python eval_vggt.py --exp_name vggt-M07gg --dataset m07 &
# CUDA_VISIBLE_DEVICES=7 python eval_vggt.py --exp_name vggt-S01gg-final --dataset m07 &
# CUDA_VISIBLE_DEVICES=4 python eval_relpose.py --exp_name spider-warp-S01gg-final --dataset m07 &
# CUDA_VISIBLE_DEVICES=5 python eval_mast3r.py --exp_name aerial-mast3r-S01gg-final --model_name aerial-mast3r --dataset m07 &
# CUDA_VISIBLE_DEVICES=6 python eval_mast3r.py --exp_name mast3r-S01gg-final --model_name mast3r --dataset m07 &

# CUDA_VISIBLE_DEVICES=1 python eval_relpose.py --exp_name spider-region512 --dataset mega_concat &
# CUDA_VISIBLE_DEVICES=2 python eval_relpose.py --exp_name spider-region512 --dataset scannet_concat &
# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py --exp_name spider-region512 --dataset aerial_concat &
# CUDA_VISIBLE_DEVICES=0 python eval_PasF.py --exp_name spider-PasF512 --dataset mega1500 &
# CUDA_VISIBLE_DEVICES=1 python eval_PasF.py --exp_name spider-PasF512 --dataset scannet &
# CUDA_VISIBLE_DEVICES=3 python eval_PasF.py --exp_name spider-PasF512 --dataset aerial &

CUDA_VISIBLE_DEVICES=0 python eval_spiderfm.py --dataset mega1500 &
CUDA_VISIBLE_DEVICES=1 python eval_spiderfm.py --dataset scannet &
CUDA_VISIBLE_DEVICES=2 python eval_spiderfm.py --dataset aerial &
wait
echo "All scripts have finished running."
