import itertools
import os
import subprocess
from typing import List, Union

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf, ListConfig

from ner.helpers import get_dir, generate_random_string, chunks


load_dotenv()

ROOT_DIR = os.getenv('ROOT_DIR')
CONDA_ENV_NAME = os.getenv('CONDA_ENV_NAME')
CONDA_HOME = os.getenv('CONDA_HOME')
SLURM_LOG_DIR = os.getenv('SLURM_LOG_DIR')
SLURM_DIR = os.getenv('SLURM_DIR')

def main(
        model: str,
        experiment: str,
        yaml_sweep_file: str,
        exclude: Union[str, List[str]] = None,
        include: Union[str, List[str]] = None,
        partition_name: str = 'batch',
        max_runs_per_scripts: int = 1,
        use_slurm: bool = False,
):
    """This creates a slurm file and runs it
        Args:
            partition_name (str): Partition to run the code on
            model (str): model to run
            experiment (str): experiment to run
            exclude (str): nodes to exclude
            include (str): nodes to include
            yaml_sweep_file (str): file path containing the parameter to sweep through
            max_runs_per_scripts (int): maximum number of python call torun an experiment bet bash file ran
            use_slurm (bool): is this script submitted to slurm. If yes? add '_slurm' ssuffix to prevent commit and
                              call 'sbatch' else call 'bash'
        """

    # create arrays of arguments command for the sweep
    suffix = "_slurm" if use_slurm else ""
    # sweep = OmegaConf.load(f"{ROOT_DIR}/exps/{experiment}/{yaml_sweep_file}")
    sweep = OmegaConf.load(f"{ROOT_DIR}/exps/{yaml_sweep_file}")
    keys, values = zip(*sweep.items())
    arguments_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # generate a command line run for each combination of hyperparameters obtained from the sweep yaml file
    commands = []
    for arguments_chunk in chunks(arguments_list, max_runs_per_scripts):
        print(arguments_chunk, len(arguments_chunk))
        c = arguments_chunk[0]
        assert len(arguments_chunk) == 1
        # ${data.method}_${data.param}_${data.language}_${model.conf.name}
        # experiment_name = f"{c['cfg.data.method']}_{c['cfg.data.param']}_{c['cfg.data.language']}_{c['cfg.model.conf'][2]}_{c['device.seed']}"
        experiment_name = f"{c['data.method']}_{c['data.param']}_{c['data.language']}_{c['model.conf'][2]}_{c['device.seed']}"
        print(experiment_name)
        experiment = experiment_name
        command = ""
        for arguments in arguments_chunk:
            run = f"python {ROOT_DIR}/scripts/train_eval_ner.py --config-path=../exps --config-name=base"
            for key, value in arguments.items():
                run += f" {key}="
                if isinstance(value, (list, ListConfig)):
                    # we need a special care for list values as hydra
                    # will throw an error if there are quotes in the list items
                    run += "["
                    for idx in range(len(value)):
                        if idx == 0:
                            run += f"{value[idx]}"
                        else:
                            run += f",{value[idx]}"
                    run += "]"
                else:
                    run += f"{value}"
            command += f"\n{run}"
        commands.append((experiment, command))

    # Create Slurm File
    def get_bash_text(bsh_cmd, experiment):
        return f'''#!/bin/bash
#SBATCH -p {partition_name}
#SBATCH -N 1
#SBATCH -t 72:00:00
{f"#SBATCH -x {exclude if isinstance(exclude, str) else ','.join(exclude)}" if isinstance(exclude, (str, list, tuple)) else ""} 
{f"#SBATCH -w {include if isinstance(include, str) else ','.join(include)}" if isinstance(include, (str, list, tuple)) else ""}
#SBATCH -J {experiment}_{model}
#SBATCH -o {get_dir(f"{ROOT_DIR}/{SLURM_LOG_DIR}", experiment, "outputs_slurm")}/{model}.%N.%j.out
#SBATCH -e {get_dir(f"{ROOT_DIR}/{SLURM_LOG_DIR}", experiment, "errors_slurm")}/{model}.%N.%j.err
{f"source ~/.bashrc && conda activate {CONDA_ENV_NAME}" if  use_slurm else ""}
cd {ROOT_DIR}
export PYTHONPATH=$PYTHONPATH:`pwd`
{bsh_cmd}
{"conda deactivate" if  use_slurm else ""}
'''

    directory = get_dir(f"{ROOT_DIR}/{SLURM_DIR}", 'all')

    for experiment, cmd in commands:
        idx = generate_random_string()
        idx = ''
        fpath = os.path.join(directory, f'{model}_{experiment}_{idx}{suffix}.bash')
        with open(fpath, 'w+') as f:
            f.write(get_bash_text(cmd, experiment))

        # Run it
        if 0:
            if use_slurm:
                ans = subprocess.call(f'sbatch {fpath}'.split(" "))
            else:
                ans = subprocess.call(f"""
                source {CONDA_HOME}/etc/profile.d/conda.sh
                conda activate {CONDA_ENV_NAME}
                bash {fpath}
                """, shell=True, executable='/bin/bash')
            assert ans == 0
            print(f"Successfully called {fpath}")


if __name__ == "__main__":
    fire.Fire(main)
