#!/bin/bash
CLUSTER=${1}
JOB_SCRIPT=${2}
source _Cluster/${CLUSTER}.sh
shift 2

sbatch --account=${CLUSTER_ACCOUNT} --partition=${CLUSTER_PARTITION} \
    ${JOB_DEPENDENCY:+--dependency=${JOB_DEPENDENCY}} \
    ${JOB_TIME:+--time=${JOB_TIME}} \
    --export=ALL,CLUSTER_FILE="_Cluster/${CLUSTER}.sh" "${JOB_SCRIPT}" "$@"