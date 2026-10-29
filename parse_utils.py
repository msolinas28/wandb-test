import ast
import argparse
import jax.numpy as jnp
import yaml

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError("Boolean value expected.")
    
def dict_or_none(v):
    if v.lower() == "None":
        return None
    try:
        return ast.literal_eval(v)
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError("Expected a dictionary or 'None'.")
    
def tuple_or_none(v):
    if v.lower() == "none":
        return None
    try:
        value = ast.literal_eval(v)
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError("Expected a tuple or 'None'.")

    if not isinstance(value, tuple):
        raise argparse.ArgumentTypeError("Expected a tuple.")

    return value
    
def none_or_float(value):
    if value.lower() == "none":
        return None
    return float(value)

def none_or_str(value):
    if value.lower() == "none":
        return None
    return value

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)

def print_parsed(args):
    for key, value in vars(args).items():
        if hasattr(value, "label"):
            print(f"{key}: {value.label}")
        else:
            print(f"{key}: {value}")

mapping = {
        "c128": jnp.complex128,
        "f64": jnp.float64,
        "f32": jnp.float32,
        "f16": jnp.float16,
        "bf16": jnp.bfloat16,
    }

def str_to_jax_dtype(srt_dtype):
    try:
        return mapping[srt_dtype.lower()]
    except KeyError:
        raise NotImplementedError("Pick a valid dtype: 'c128', 'f64', 'f32', 'f16', or 'bf16'")
    
def jax_dtype_to_str(jax_dtype):
    for key, value in mapping.items():
        if value == jax_dtype:
            return key
    raise NotImplementedError("Pick a valid jax dtype: jnp.float64, jnp.float32, jnp.float16 or jnp.floatb16")

def is_jax_dtype(x):
    return x in [jnp.complex128, jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]

class jax_dtype():
    def __init__(self, dtype):
        if dtype is None or (isinstance(dtype, str) and dtype.lower() == "none"):
            self.label = None
            self.dtype = None
        elif isinstance(dtype, str):
            self.label = dtype
            self.dtype = str_to_jax_dtype(dtype)
        elif is_jax_dtype(dtype):
            self.label = jax_dtype_to_str(dtype)
            self.dtype = dtype
        else:
            raise NotImplementedError("dtype has to be a sting or a jax dtype.")

def read_config(args):
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if config_path.suffix != '.yaml':
            raise ValueError(f'config must point to a .yaml, got {config_path}')

        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f)
        
        args_dict = vars(args)
        for k, v in yaml_cfg.items():
            if k not in args_dict.keys():
                raise ValueError(f"Got unexpected argument {k}.")
            setattr(args, k, v)