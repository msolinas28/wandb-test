export CLUSTER_ACCOUNT=neuquass
export CLUSTER_PARTITION=develbooster
export CLUSTER_GPU_TYPE=''
export CLUSTER_TIME='06:00:00'

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cluster_init() {
    module load CUDA-Python/12
    source "$SCRIPT_DIR/$SCRATCH/env/jVMC_gpu/bin/activate"
}