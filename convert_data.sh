#!/bin/bash

#SBATCH --account=rrg-mgoubran
#SBATCH --job-name=convert_3d_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04-00:00:00
#SBATCH --output=convert_3d_data-%j.out


module load scipy-stack opencv
source venv/bin/activate

which python
pwd

python -u convert_data.py --dataset-type Tr
python -u convert_data.py --dataset-type Val
python -u convert_data.py --dataset-type Ts

