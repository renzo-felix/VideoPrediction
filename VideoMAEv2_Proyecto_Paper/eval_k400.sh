#!/bin/bash
#SBATCH --job-name=eval_k400
#SBATCH --output=logs/eval_k400_%j.out
#SBATCH --error=logs/eval_k400_%j.err
#SBATCH --partition=gpu
#SBATCH --account=a-investigacion1
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
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path dataset/k400_luis/labels/val.csv \
    --data_root dataset/k400_luis/k400 \
    --finetune checkpoints/vit_g_hybrid_pt_1200e_k400_ft.pth \
    --log_dir output_dir/k400_eval \
    --batch_size 4 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --sampling_rate 4 \
    --eval \
    --dist_eval
