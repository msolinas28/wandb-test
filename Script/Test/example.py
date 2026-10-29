import jVMC_exp
import jVMC_exp.operator.discrete as op
import jax
import argparse
import wandb

import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = None
for parent in current_path.parents:
    if (parent / ".project_root").exists():
        project_root = parent
        break
if project_root is None:
    raise RuntimeError(
        "Could not find the project root. Make sure a '.project_root' file exists."
    )
sys.path.append(str(project_root))
from parse_utils import (
    print_parsed, read_config, 
    str2bool, none_or_int, none_or_str
)
from project_paths import PROJECT_ROOT
from file_handler import HistoryHandler, ParHandler
from wandb_util import init_wandb, wandb_callback_gs
from net import RBM

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=none_or_str, default=None, help="Path to YAML config")
parser.add_argument("--n_samples", default=2**12, type=int)
parser.add_argument("--n_chains", default=2**10, type=int)
parser.add_argument("--n_spins", default=15, type=int)
parser.add_argument("--batch_size", default=None, type=none_or_int)
parser.add_argument("--n_steps", default=250, type=int)
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--diag_shift", default=1e-3, type=float)
parser.add_argument("--lr", default=1e-1, type=float)
parser.add_argument("--save_history", default=True, type=str2bool)
parser.add_argument("--save_params", default=True, type=str2bool)
parser.add_argument("--J", default=-1, type=float)
parser.add_argument("--h", default=1, type=float)
parser.add_argument("--resume_run", default=False, type=str2bool)
args = parser.parse_args()
read_config(args)
is_main_process = jax.process_index() == 0
if is_main_process:
    print_parsed(args)

folder = f"{PROJECT_ROOT}/Data/"

if args.batch_size is None:
    args.batch_size = args.n_samples

model = RBM(n_hidden=args.n_spins)

# Initialize wandb
init_wandb(
    args=argparse.Namespace(**vars(args), **model.info),
    script_name=__file__, 
    group_name=model.label
)

psi = jVMC_exp.vqs.NQS(model, args.n_spins, args.batch_size, seed=args.seed)
proposer = jVMC_exp.propose.SpinFlip()
sampler = jVMC_exp.sampler.MCSampler(
    psi, 
    proposer, 
    args.seed,
    args.n_chains, 
    args.n_samples
)

hamiltonian = 0
m_x = 0
for i in range(args.n_spins):
    hamiltonian += args.J * op.SigmaZ(i) * op.SigmaZ((i + 1) % args.n_spins)
    hamiltonian -= args.h * op.SigmaX(i)
    m_x += op.SigmaX(i)
m_x /= args.n_spins
observables = dict(m_x=m_x)

loss = jVMC_exp.objective_function.Observable(hamiltonian)
solver = jVMC_exp.solver.Pinv()
stepper = jVMC_exp.stepper.Euler(args.lr)
opt = jVMC_exp.optimizer.SR(sampler, psi, args.diag_shift, solver=solver)

results = opt.ground_state_search(
    args.n_steps, 
    loss,
    stepper, 
    observables=observables,
    callback=wandb_callback_gs(opt)
)

if args.save_history and is_main_process:
    hh = HistoryHandler(
        folder, 
        "TFIM", 
        model.label, 
        args.n_spins, 
        PBC=True, 
        n_dim=1, 
        **vars(args),
        **model.info
    )
    hh.save_history(opt.output_manager)
if args.save_params and is_main_process:
    ph = ParHandler(
        folder, 
        "TFIM", 
        model.label, 
        args.n_spins, 
        PBC=True, 
        n_dim=1, 
        **vars(args),
        **model.info
    )
    ph.save(psi.parameters)

wandb.finish() # Close wandb