#!/bin/bash
#SBATCH --job-name=eval_ssv2
#SBATCH --output=logs/eval_ssv2_%j.out
#SBATCH --error=logs/eval_ssv2_%j.err
#SBATCH --partition=gpu
#SBATCH --account=investigacion1     
#SBATCH --qos=a-investigacion1
#SBATCH --time=4-00:00:00               
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate videomae_luis_izaguirre

export SLURM_NTASKS=1

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
    run_class_finetuning.py \
    --model vit_giant_patch14_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path dataset/ssv2_luis/labels/sthv2 \
    --data_root dataset/ssv2_luis \
    --finetune checkpoints/vit_g_hybrid_pt_1200e_ssv2_ft.pth \
    --log_dir output_dir/ssv2_eval \
    --batch_size 4 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --sampling_rate 4 \
    --eval \
    --dist_eval