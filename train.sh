#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --partition=a100
#SBATCH --account=bbym-hydro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --output=/projects/bbym/nathanj/ML-proj/imagenet_and_augment.out

# Load required modules
module load anaconda3_gpu cudnn

# Change directory
cd /projects/bbym/nathanj/attentionUnet

# Execute the python script
python train.py --name_id 'Alexander_train'