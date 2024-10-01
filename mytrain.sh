#!/bin/bash

#SBATCH --account=rrg-mgoubran
#SBATCH --job-name=train-medsam-slice-official
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --time=06-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=train-medsam-slice-2gpu%j.out


module load scipy-stack opencv cuda
source venv/bin/activate

which python
pwd

python -u train_3d.py \
    -exp_name medsam-slice-192img-16video \
    -vis True \
    -pretrain "checkpoints/MedSAM2_pretrain.pth" \
    -val_freq 1 \
    -gpu True \
    -gpu_device 0 \
    -image_size 192 \
    -out_size 192 \
    -dataset neurosam \
    -sam_ckpt "checkpoints/MedSAM2_pretrain.pth" \
    -video_length 16 \
    -b 2 \
    -data_path "/home/arinaldi/projects/rrg-mgoubran/arinaldi/SliceData" \
    -sam_config sam2_hiera_t_smaller_image
