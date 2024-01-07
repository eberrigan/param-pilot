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