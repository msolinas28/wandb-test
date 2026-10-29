#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=_Logs/%j.out

source ${CLUSTER_FILE}
cluster_init

if [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then
    export JVMC_USE_DISTRIBUTED=true
else
    export JVMC_USE_DISTRIBUTED=false
fi

python Test/example.py