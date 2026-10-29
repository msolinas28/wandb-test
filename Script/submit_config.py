import os
import subprocess
from pathlib import Path
import argparse

import sys
current_path = Path(__file__).resolve()
project_root = None
for parent in current_path.parents:
    if (parent / ".project_root").exists():
        project_root = parent
        break
if project_root is None:
    raise RuntimeError("Could not find the project root. Make sure a '.project_root' file exists.")
sys.path.append(str(project_root))
from project_paths import PROJECT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("cluster")
parser.add_argument("script")
parser.add_argument("config")
parser.add_argument("--dependency", default=None,
                    help="SLURM dependency, e.g. afterok:123456 (start only if 123456 succeeds) or afterany:123456")
parser.add_argument("--time", default=None,
                    help="SLURM time limit, e.g. 12:00:00; overrides the #SBATCH --time of the job script (default 04:00:00)")
args = parser.parse_args()

config = Path(args.config)
if config.parent == Path('.'):
    config = Path(f'{PROJECT_ROOT}/Script/_Config/{args.config}')

if config.suffix == '':
    configs = sorted(config.rglob("*.yaml"))
    if not configs:
        raise FileNotFoundError(
            f"No .yaml configs found in: {config}"
        )
elif config.suffix == '.yaml':
    if not config.is_file():
        raise FileNotFoundError(f"Config file not found: {config}")
    configs = (config,)
else:
    raise ValueError(f'For config give a folder or a .yaml, got {args.config}')
print(f"Found {len(configs)} configs. Submitting to cluster '{args.cluster}'...")

env = os.environ.copy()
env.pop("JOB_DEPENDENCY", None)
if args.dependency:
    env["JOB_DEPENDENCY"] = args.dependency
env.pop("JOB_TIME", None)
if args.time:
    env["JOB_TIME"] = args.time

for cfg in configs:
    cmd = ["./_sbatch.sh", args.cluster, "_send_config.sh", args.script, str(cfg)]
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"Failed to submit {cfg}")
    else:
        print(f"Submitted {cfg}")