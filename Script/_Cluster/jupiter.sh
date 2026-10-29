export CLUSTER_ACCOUNT=neuquass
export CLUSTER_PARTITION=booster
export CLUSTER_GPU_TYPE=''
export CLUSTER_TIME='06:00:00'
export CLUSTER_SYNC_INTERVAL=120

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRATCH=/e/scratch/neuquass/solinas1

cluster_init() {
    export WANDB_MODE=offline

    module --force purge
    module load Stages/2026
    module load GCCcore/.14.3.0
    module load CUDA/13
    module load Python/3.13.5
    module load nano
    module load cuDNN/9.19.0.56-CUDA-13
    module load NCCL/default-CUDA-13
    source "$SCRATCH"/env/jVMC/bin/activate
}

cluster_watch_sync() {
    local wandb_dir="$SCRATCH/Graphene_Time_Evolution/Data/wandb"

    module --force purge
    module load Stages/2026
    module load GCCcore/.14.3.0
    module load Python/3.13.5
    source "$SCRATCH"/env/jVMC/bin/activate

    sync_job() {
        local j=$1
        local f
        f=$(grep -laP "job_id.{0,10}\\b${j}\\b" "${wandb_dir}"/offline-run-*/run-*.wandb 2>/dev/null | head -1)
        [[ -n "${f}" ]] || return 0
        local run_dir
        run_dir=$(dirname "${f}")
        echo "=== $(date -Is) job ${j} ==="
        wandb sync --yes --no-skip-synced --include-offline "${run_dir}"
    }

    local prev_jobs=()
    while true; do
        local squeue_output squeue_rc
        squeue_output=$(squeue -u "$(whoami)" -h -o "%A" 2>/dev/null)
        squeue_rc=$?

        local current_jobs=()
        if [[ ${squeue_rc} -eq 0 ]]; then
            mapfile -t current_jobs <<< "${squeue_output}"
            [[ ${#current_jobs[@]} -eq 1 && -z "${current_jobs[0]}" ]] && current_jobs=()
        else
            current_jobs=("${prev_jobs[@]}")
        fi

        local j
        for j in "${current_jobs[@]}"; do
            sync_job "${j}"
        done
        for j in "${prev_jobs[@]}"; do
            if [[ ! " ${current_jobs[*]} " == *" ${j} "* ]]; then
                sync_job "${j}"
            fi
        done

        [[ ${#current_jobs[@]} -eq 0 ]] && break
        prev_jobs=("${current_jobs[@]}")
        sleep "${CLUSTER_SYNC_INTERVAL}"
    done
    echo "No active jobs left -- exiting."
}