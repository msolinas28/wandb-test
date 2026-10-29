import hashlib, json, os
import wandb
import jax
from jVMC_exp.optimizer.base import AbstractOptimizer

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
from project_paths import PROJECT_ROOT, PROJECT_NAME

def init_wandb(
          args,
          script_name: str,
          group_name: str = None,
          project_name: str = PROJECT_NAME,
          resume: bool = False
):
    script_name = os.path.basename(script_name)
    args_dict = vars(args)
    tag = [Path(args.config).parent.name if args.config else "manual_runs"]
    s = json.dumps({"script": script_name, **args_dict}, sort_keys=True, default=str)
    run_id = hashlib.md5(s.encode()).hexdigest()[:10]

    wandb_mode = os.environ.get('WANDB_MODE', 'online')
    if jax.process_index() != 0:
        wandb_mode = "disabled"
    else:
        print('wandb mode: ', wandb_mode)

    attempt = 0
    while True:
        id_ = run_id if attempt == 0 else f"{run_id}-{attempt}"
        try:
            wandb.init(
                mode=wandb_mode,
                project=project_name,
                dir=f"{PROJECT_ROOT}/Data/",
                id=id_,
                resume="allow" if resume else "never",
                config=args_dict,
                group=group_name,
                tags=tag,
                job_type=script_name,
                name=Path(args.config).stem if args.config else f"manual_{id_}",
            )
            return
        except wandb.errors.CommError as e:
            if "previously created and deleted" not in str(e):
                raise
            attempt += 1

def wandb_callback_gs(opt: AbstractOptimizer):
    def callback():
        log = {}

        for k, v in opt.output_manager.data['observables'].items():
            if k == 'times':
                log['step'] = v[-1]
            else:
                log[k + '_mean'] = v['mean'][-1].item()
                log[k + '_variance'] = v['variance'][-1].item()
                log[k + '_MC_error'] = v['MC_error'][-1].item()

        log['R_hat'] = opt.o_loc.get_R_hat(opt.sampler.numChains)

        wandb.log(log)
    
    return callback