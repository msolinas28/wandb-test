import pandas as pd
import os
import json
from flax import serialization
import jax.numpy as jnp
import jax

from jVMC_exp.util.output_manager import OutputManager

class Handler:
    def __init__(self, folder: str, model: str, arch: str, n_particles: int, PBC: bool, n_dim: int, **kwargs):
        self.folder = folder + self._to_save_label.capitalize() + "/" + model.capitalize() + "/"
        self.model = model
        self.arch = arch
        self.n_particles = n_particles
        self.PBC = PBC
        self.n_dim = n_dim
        self.kwargs = dict(sorted(kwargs.items()))

        if PBC: 
            self.BC = "PBC"
        else: 
            self.BC = "OBC"

    @property
    def list_name(self):
        return f"{self.arch}.csv"
    
    @property
    def _f_list_name(self):
        return self.folder + self.list_name
    
    @property
    def _exist_list(self):
        return os.path.exists(self._f_list_name)
    
    @property 
    def _keys(self):
        ks = ["BC", "n_dim", "n_particles"]
        if self.kwargs is not None:
            for k in list(self.kwargs.keys()):
                ks.append(k)
        return ks
    
    @property
    def _values(self):
        vs = [self.BC, self.n_dim, self.n_particles]
        if self.kwargs is not None:
            for v in list(self.kwargs.values()):
                vs.append(v)
        return vs
    
    @property
    def _dict_to_add(self):
        _dict_to_add = dict(zip(self._keys, self._values))
        return dict(sorted(_dict_to_add.items()))
    
    def exist_in_list(self):
        if self._exist_list:
            mask = self.df_list["tags"].apply(lambda x: x == self._dict_to_add)
            if sum(list(mask)):
                return list(self.df_list[mask].index)[0] + 1
            return 0
        else:
            raise ValueError(f"The following file does not exist: {self._f_list_name}")
        
    def _append_in_list(self):
        df = pd.concat([self.df_list, self._df_to_add], ignore_index=True)
        df['tags'] = df['tags'].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else x)
        df.to_csv(self._f_list_name, index=False)
            
    def _save_index(self):
        if self._exist_list:
            index = self.exist_in_list()
            if not index: 
                self._append_in_list()
                index = len(self.df_list)
            print("List updated!")
        else:
            self._create_list()
            index = 1
        return index

    def _load_index(self):
        index = self.exist_in_list()
        if not index:
            msg = "Parameters:\n" + "\n".join(f"    {k}: {v}" for k, v in self.kwargs.items())
            raise ValueError(
                    f"The selected {self._to_save_label} do not exist!\n"
                    f"  Architecture:      {self.arch}\n"
                    f"  Model:      {self.model}\n"
                    f"  {msg}\n"
                    f"  n_particles:    {self.n_particles}\n"
                    f"  n_dim:      {self.n_dim}\n"
                    f"  BC:         {self.BC}\n"
                    f"  Folder: {self.folder}"
                )
        return index

    def _create_list(self):
        if self._exist_list:
            raise ValueError("The list already exists!")
        
        os.makedirs(self.folder, exist_ok=True)
        self.df_list.to_csv(self._f_list_name, index=False)
        print("List Created!")
    
    @property
    def _df_to_add(self):
        return pd.DataFrame([{"tags": json.dumps(self._dict_to_add, sort_keys=True)}])

    @property
    def df_list(self):
        if self._exist_list:
            df = pd.read_csv(self._f_list_name)
            df["tags"] = df["tags"].apply(json.loads)
            return df
        else:
            return self._df_to_add
    
    @property
    def _to_save_label(self) -> str:
        return NotImplementedError
    
class ParHandler(Handler):
    @property
    def _to_save_label(self):
        return "parameters"

    @property
    def _parameters_folder(self):
        return self.folder + self.arch.capitalize() + "/"

    def save(self, parameters):
        index = self._save_index()
        binary_data = serialization.to_bytes(parameters)
        os.makedirs(self._parameters_folder, exist_ok=True)
        with open(f"{self._parameters_folder}{str(index)}.mpack", "wb") as outfile:
            outfile.write(binary_data)
        print("Parameters saved!")
    
    def load(self, parameters):
        with open(self.parameters_file, "rb") as infile:
            binary_data = infile.read()
            parameters = serialization.from_bytes(parameters, binary_data)
            parameters = jax.tree.map(lambda x: jnp.array(x), parameters)

        return parameters
    
    @property
    def parameters_file(self):
        index = self._load_index()
        str_index = str(index)
        folder = self._parameters_folder

        return f"{folder}{str_index}.mpack"
    
class HistoryHandler(Handler):
    @property
    def _to_save_label(self):
        return "history"
    
    @property
    def history_file(self):
        index = self._load_index()
        str_index = str(index)
        return f"{self.folder}{self.arch.capitalize()}/{str_index}.h5"
    
    def save_history(self, output_manager: OutputManager, append=False):
        index = self._save_index()
        folder = self.folder + self.arch.capitalize() + "/"
        os.makedirs(folder, exist_ok=True)
        filepath = f"{folder}{str(index)}.h5"
        output_manager.save_to_h5(filepath, append)

        print("History saved.")

    def load_history(self):
        return OutputManager().load_from_h5(self.history_file)