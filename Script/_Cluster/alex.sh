export CLUSTER_ACCOUNT=v166da
export CLUSTER_PARTITION='a100'
export CLUSTER_GPU_TYPE='a100'
export CLUSTER_TIME='01:00:00'

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cluster_init() {
    export WANDB_MODE=online
    export http_proxy=http://proxy.nhr.fau.de:80
    export https_proxy=http://proxy.nhr.fau.de:80

    module purge
    module load python/3.12-base 
    module load cuda/13.2.0
    module load cudnn/9.17.0.29-13

    CUDA_BASE=/apps/SPACK/1.1.1-alex/opt/spack/linux-almalinux9-zen3/none-none/cuda-13.2.0-muu4y5j66hnxz4ijk3xkmchxguoddbsh
    export LD_LIBRARY_PATH=${CUDA_BASE}/targets/x86_64-linux/lib:${CUDA_BASE}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

    # NCCL for multi-GPU
    export LD_LIBRARY_PATH=/apps/jupyterhub/jh5.4.4-py3.12/envs/pytorch-2.10.0/lib:$LD_LIBRARY_PATH
    source "$SCRIPT_DIR/../../../env/jVMC_new/bin/activate"
}