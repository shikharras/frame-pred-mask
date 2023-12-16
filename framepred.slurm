#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_001-2023fa-16

### change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --time=1-23:00:00
#SBATCH --output=ffp_%j.out
#SBATCH --error=ffp_%j.err

singularity exec --nv --bind /scratch --overlay /scratch/sr6644/overlay-25GB-500K.ext3:ro /scratch/sr6644/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "cd /scratch/sr6644/frame-pred-mask; source /ext3/miniconda3/etc/profile.d/conda.sh ; conda activate dl-proj; python next_frame_pred_lightning.py"