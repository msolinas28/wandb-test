#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 # Put this to one if n_nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=_Logs/%j.out

SCRIPT=${1}
CONFIG=${2}

source ${CLUSTER_FILE}
cluster_init

if [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then
    export JVMC_USE_DISTRIBUTED=true
    srun python "${SCRIPT}" --config="${CONFIG}"
else
    export JVMC_USE_DISTRIBUTED=false
    python "${SCRIPT}" --config="${CONFIG}"
fi