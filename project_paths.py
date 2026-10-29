from pathlib import Path

PROJECT_NAME = "wandb-test"

def find_project_root(target=PROJECT_NAME, start_path=None):
    """
    Returns the absolute path of the folder named `target`
    starting from `start_path` (default: current working directory).
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()

    current = start_path

    while current.name != target:
        if current.parent == current:
            raise FileNotFoundError(f"Could not find a folder named '{target}' in any parent directories.")
        current = current.parent

    return str(current)

PROJECT_ROOT = find_project_root()
