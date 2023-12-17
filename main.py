import os
import subprocess as sp

gpu = "..." #number of GPUS goes here (can be fractional)
job_name = "..." #job name goes here

base = "/home/runner/talmodata-smb/path/to/cwd"
repo = base.replace("path/to/cwd", "path/to/repo/to/install") #may be unnecessary if you arent testing a local package

container = "..." #docker container goes here (ie "arlo/biogtr")
env = "..." #conda environment name (must be preinstalled on docker container)

run_file_p = ... #train script goes here (see above)
run_file_p = os.path.join(base, run_file_p)

pods = ... #number of tasks needed to be run
par = ... #number of tasks to run simultaneously (par <= pods)

cmd = [
    "runai",
    "submit",
    "--gpu",
    gpu,
    "--name",
    job_name,
    "--preemptible",
    "-i",
    container,
    "-v",
    "/data/talmolab-smb:/home/runner/talmodata-smb",
    "-e",
    f"RUNNER_CMD=cp -r {repo} ~ && pip install --user -e ~/repo_name && conda run -n {env} python {run_file_p}",
    "--parallelism",
    str(par),
    "--completions",
    str(pods),
]

print(f"base directory: {base}")
print(f"running with {pods} pods")
print(f"max pods that can run concurrently: {par}")
print(f"runner command: {cmd}")

def create_hyperparam_grid(hp):
    combs = list(itertools.product(*hp.values()))
    df = pd.DataFrame(combs, columns=hp.keys())
    return df


def get_hyperparams(df, index):
    hparams = df.iloc[index].to_dict()

    return hparams

def modify_hyperparams(hparams, params_config):
    #TODO: Modify hyper_parameters

    return hparams

if __name__ == "__main__":

    hp = {
        "{HPARAM1}": [VALS_TO_GRID_SEARCH],
        "{HPARAM2}": [VALS_TO_GRID_SEARCH],
        ...
    }

    # get pod index
    try:
        batch = True
        index = int(os.environ["POD_INDEX"])
    except Exception as e:
        print("No pod index found! Assuming Single Run")
        index = 0
        batch = False

    if batch:
        base = "/home/runner/path/to/cwd/"
        data_path = (
            "/home/runner/path/to/data"
        )
    else:
        base = "/{local_home_dir}/path/to/cwd"
        data_path = (
            "{local_home_dir}/path/to/data"
        )
    df = create_hyperparam_grid(hp)
    hparams = get_hyperparams(df, index=index)
    hparams = modify_hyperparams(hparams, params_config)

    #Training logic goes here with modified hyperparameters
