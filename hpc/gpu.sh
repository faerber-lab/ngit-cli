#!/bin/bash
#SBATCH --job-name=ngit-cli
#SBATCH --output=joaa_gpu_output_%j.log
#SBATCH --error=joaa_gpu_error_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohimenul.joaa@gmail.com

module load CUDA/12.4  # CUDA version compatible with torch 2.5.1
module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
source ~/.bashrc

#conda env list
#conda list
#which python

conda activate /data/horse/ws/afjo837h-ngit-ws/py310sep4

JOBID="$SLURM_JOB_ID"

# Run your script
/data/horse/ws/afjo837h-ngit-ws/py310sep4/bin/python tests/smallest_model_test.py --job-id "$JOBID"
