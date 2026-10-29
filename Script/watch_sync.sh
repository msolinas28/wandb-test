#!/bin/bash
CLUSTER=${1:-jupiter}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
source "_Cluster/${CLUSTER}.sh"

if ! declare -F cluster_watch_sync > /dev/null; then
    echo "Cluster '${CLUSTER}' has no watcher defined (only needed for offline-wandb clusters)."
    exit 1
fi

cluster_watch_sync