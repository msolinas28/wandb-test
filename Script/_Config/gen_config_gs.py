import yaml
from pathlib import Path

import sys
from pathlib import Path
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

folder_name = 'lr_sweep'

fixed = dict(
    n_steps=500,
    n_spins=15,
    n_samples=2**12,
    n_chains=2**10,
    batch_size=None,
    seed=123,
    diag_shift=1e-3,
    J=-1,
    h=1,
    save_history=False,
    save_params=False,
)

grid = dict(
    lr=[1e-1, 1e-2, 1e-3]
)
labels = [f"lr_{lr}" for lr in grid['lr']]

if __name__ == "__main__":
    folder_general = f'{PROJECT_ROOT}/Script/_Config/'
    if grid is None:
        with open(f"{folder_general}run.yaml", "w") as f:
            yaml.dump(fixed, f, sort_keys=False)

        print(f"Generated 1 config in {folder_general}")
    else:
        out_dir = Path(f'{folder_general}{folder_name}')
        out_dir.mkdir(parents=True, exist_ok=True)

        for old_file in out_dir.glob("*.yaml"):
            old_file.unlink()

        for i, combo in enumerate(zip(*grid.values())):
            scan_values = dict(zip(grid.keys(), combo))
            cfg = {**fixed, **scan_values}

            name = f"run{i:03d}" if labels is None else labels[i]
            with open(out_dir / f"{name}.yaml", "w") as f:
                yaml.dump(cfg, f, sort_keys=False)

        print(f"Generated {i + 1} configs in {out_dir}")
