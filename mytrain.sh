#!/bin/bash

#SBATCH --account=rrg-mgoubran
#SBATCH --job-name=train-medsam-slice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --time=00-00:20:00
#SBATCH --gres=gpu:1
#SBATCH --output=train-medsam-slice-%j.out


module load scipy-stack opencv cuda
source venv/bin/activate

which python
pwd

python -u train_3d.py \
    -exp_name medsam-slice \
    -vis True \
    -pretrain "checkpoints/MedSAM2_pretrain.pth" \
    -val_freq 1 \
    -gpu True \
    -gpu_device 0 \
    -image_size 512 \
    -out_size 512 \
    -dataset neurosam \
    -sam_ckpt "checkpoints/MedSAM2_pretrain.pth" \
    -video_length 30 \
    -b 2 \
    -data_path "/home/arinaldi/projects/rrg-mgoubran/arinaldi/SliceData" \
    -sam_config sam2_hiera_t
