# Script folder guide

How to launch training jobs on the different clusters, and how to monitor them once they're running.

## Layout

| Path | What it is |
|---|---|
| `submit_config.py` | Main entry point — submits one SLURM job per config file. |
| `send_job.sh` | Ad-hoc SLURM job script for one-off/manual runs. Edit the `python ...` line directly, then submit it yourself. |
| `_sbatch.sh` | Internal — sources a cluster profile and calls `sbatch`. Not called directly. |
| `_send_config.sh` | Internal — the actual SLURM job script `submit_config.py` submits (runs `python <script> --config=<config>`). |
| `_Cluster/<name>.sh` | Per-cluster profile: SLURM account/partition, module loads, venv activation, and (JUPITER only) the wandb watcher. |
| `_Config/` | YAML configs, organized in named sweep folders (e.g. `E_c_sweep/`). |
| `_Logs/` | One `<jobid>.out` per submitted job (SLURM stdout+stderr), plus the wandb sync log on JUPITER. |
| `watch_sync.sh` | Manual wandb sync watcher for JUPITER (see [Monitoring](#monitoring)). |

## Launching jobs

### 1. Config-based (recommended)

```bash
python submit_config.py <cluster> <script> <config>
```

- `<cluster>`: `jupiter`, `juwels`, or `alex`.
- `<script>`: path to a training script relative to `Script/`, e.g. `Quantum_dot/interacting_gs.py`.
- `<config>`: how it's resolved depends on whether you give a bare name or a path:
  - A **bare name** (no `/`) is looked up directly under `_Config/`. Point it at a single `.yaml` file, or at a folder — every `.yaml` found inside (recursively) gets its own job.
  - A **path containing `/`** is used as-is relative to `Script/` (or give an absolute path). Needed when you want a specific file inside a subfolder rather than the whole sweep.

Examples (run from `Script/`):
```bash
python submit_config.py jupiter Quantum_dot/interacting_gs.py E_c_sweep                    # whole sweep, one job per yaml
python submit_config.py jupiter Quantum_dot/interacting_gs.py _Config/E_c_sweep/E_c_11.yaml  # just this one config
```

To hold the new job(s) until another job finishes, add `--dependency` (passed straight to `sbatch --dependency`):
```bash
python submit_config.py jupiter Quantum_dot/interacting_gs.py E_c_sweep --dependency=afterok:123456   # only if 123456 succeeds
python submit_config.py jupiter Quantum_dot/interacting_gs.py E_c_sweep --dependency=afterany:123456  # whenever 123456 ends, success or not
```
With `afterok`, if the earlier job fails or times out the new one sits in the queue forever as `DependencyNeverSatisfied`; `scancel` it. For the ad-hoc path, set the env var instead: `JOB_DEPENDENCY=afterok:123456 ./_sbatch.sh <cluster> send_job.sh`.

To generate a new sweep of configs, see `_Config/gen_config_gs.py` / `gen_config_tdvp.py` — edit the fixed/swept parameters at the top and run them; they write the resulting `.yaml` files into a new `_Config/<folder_name>/`.

### 2. Ad-hoc / manual runs

For a quick one-off run with hand-picked CLI args instead of a config file, edit the `python ...` line in `send_job.sh` (uncomment/adjust one of the example blocks, or write your own), then submit it directly:

```bash
./_sbatch.sh <cluster> send_job.sh
```

### Resource requests

`--nodes`, `--gres=gpu:...`, `--cpus-per-task`, and `--time` are `#SBATCH` directives at the top of `_send_config.sh` (config-based path) and `send_job.sh` (ad-hoc path). Edit those directly if a job needs more time or a different GPU count.

### Clusters

| Cluster | Notes |
|---|---|
| `jupiter` | GH200 nodes, `booster` partition. **Compute nodes have no internet**, so wandb runs offline and must be synced from the login node — see Monitoring below. |
| `juwels` | `develbooster` partition, wandb online as normal. |
| `alex` | FAU Erlangen `a100` partition, wandb online via `proxy.nhr.fau.de`. |

## Monitoring

### Job status

```bash
squeue -u $USER
sacct -u $USER -X --format=JobID,JobName,State,ExitCode,Start,End   # recent history + exit codes
```

Each job's stdout/stderr is in `_Logs/<jobid>.out`.

### wandb sync (JUPITER only)

Offline runs only reach wandb.ai once something on the login node syncs them. Run the watcher manually whenever you want live progress while a job is running:

```bash
./watch_sync.sh            # defaults to jupiter; pass a cluster name to override
```

It polls `squeue` for your active jobs, syncs each one's run as it goes, does one more sync the instant a job finishes, and exits on its own once none of your jobs are left in the queue. It runs in the foreground — use `tmux`/`screen`, or `nohup ./watch_sync.sh &`, if it needs to survive a dropped SSH session.

Check whether one is already running:
```bash
pgrep -af '[w]atch_sync.sh'
```

**Convenience `.bashrc` helpers** — add these to `~/.bashrc` to manage the watcher without typing the above by hand:

```bash
is_watcher_running() {
	local pid
	pid=$(pgrep -f '[w]atch_sync.sh' | head -1)
	if [[ -n "${pid}" ]]; then
		echo "True ${pid}"
	else
		echo "False"
	fi
}

send_watcher() {
	local status
	status=$(is_watcher_running)
	if [[ "${status}" == True* ]]; then
		echo "A watcher is already running PID=${status#True }"
		return 1
	fi

	local out=~/msolinas/Graphene_Time_Evolution/Script/_Logs/_watcher.log
	nohup ~/msolinas/Graphene_Time_Evolution/Script/watch_sync.sh > "${out}" 2>&1 &
	disown
}
```

`is_watcher_running` prints `True <PID>` or `False`. `send_watcher` refuses to start a second watcher if one's already running, otherwise launches it detached (survives a dropped SSH session) with output going to `_Logs/_watcher.log` (overwritten fresh on each launch).

To force-sync one specific run by hand (e.g. one already marked synced that you want refreshed):
```bash
wandb sync --yes --no-skip-synced --include-offline Data/wandb/<run-directory>
```
