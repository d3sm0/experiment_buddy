import datetime
import logging
import os
import re
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import cloudpickle
import fabric
import git
import matplotlib.pyplot as plt
import tqdm
import wandb
import wandb.cli
import yaml
from invoke import UnexpectedExit
from paramiko.ssh_exception import SSHException

import experiment_buddy.utils

try:
    from orion.client import build_experiment
except ImportError:
    ORION_ENABLED = False
else:
    ORION_ENABLED = True

try:
    import torch
except ImportError:
    TORCH_ENABLED = False
    torch = None
else:
    TORCH_ENABLED = True

logging.basicConfig(level=logging.DEBUG)

tb = tensorboard = None
if os.path.exists("buddy_scripts/"):
    SCRIPTS_PATH = "buddy_scripts/"
else:
    SCRIPTS_PATH = os.path.join(os.path.dirname(__file__), "../scripts/")
ARTIFACTS_PATH = "runs/"
DEFAULT_WANDB_KEY = os.path.join(os.environ["HOME"], ".netrc")


def _is_running_on_cluster():
    return "SLURM_JOB_ID" in os.environ.keys() or "BUDDY_IS_DEPLOYED" in os.environ.keys()


class WandbWrapper:
    def __init__(self, wandb_kwargs, debug: bool = False, local_tensorboard=None):
        """
        :param wandb_kwargs: kwargs to pass to wandb.init
        :param debug: if True, wandb will not be used
        :param local_tensorboard: if not None, will be used to log to tensorboard
        """
        # Calling wandb.method is equivalent to calling self.run.method
        # I'd rather to keep explicit tracking of which run this object is following
        wandb_kwargs["mode"] = wandb_kwargs.get("mode", "disabled" if debug else "online")
        if not debug:
            # TODO: fork is problematic for torch distributed. Can we use thread?
            wandb_kwargs["settings"] = wandb_kwargs.get("settings", wandb.Settings(start_method="fork"))
        self.run = wandb.init(**wandb_kwargs)
        # TODO: we can remove wandb all together and use tensorboard with track tensorboard
        self.tensorboard = local_tensorboard
        # TODO: this is now being taken care by hydra
        self.objects_path = os.path.join(ARTIFACTS_PATH, "objects/", self.run.name)
        os.makedirs(self.objects_path, exist_ok=True)

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        if scalar_value != scalar_value:
            warnings.warn(f"{tag} is {scalar_value} at {global_step} :(")

        scalar_value = float(scalar_value)  # silently remove extra data such as torch gradients
        self.run.log({tag: scalar_value}, step=global_step, commit=False)
        if self.tensorboard:
            self.tensorboard.add_scalar(tag, scalar_value, global_step=global_step)

    def add_scalars(self, dict_of_scalars: Dict[str, float], global_step: int = None, prefix: str = ""):
        for k, v in dict_of_scalars.items():
            self.add_scalar(prefix + k, v, global_step)

    def add_figure(self, tag, figure, global_step=None, close=True):
        self.run.log({tag: figure}, global_step)
        if close:
            plt.close(figure)

        if self.tensorboard:
            self.tensorboard.add_figure(tag, figure, global_step=None, close=True)

    @staticmethod
    def add_histogram(tag, values, global_step=None):
        if len(values) <= 2:
            raise ValueError("histogram requires at least 3 values")

        if isinstance(values, (tuple, list)) and len(values) == 2:
            wandb.log({tag: wandb.Histogram(np_histogram=values)}, step=global_step, commit=False)
        else:
            wandb.log({tag: wandb.Histogram(values)}, step=global_step, commit=False)

    def plot(self, tag, values, global_step=None):
        wandb.log({tag: wandb.Image(values)}, step=global_step, commit=False)
        plt.close()

    def add_object(self, tag, obj, global_step=None):
        if not TORCH_ENABLED:
            raise NotImplementedError

        local_path = os.path.join(self.run.dir, f"{tag}-{global_step}.pt")
        with open(local_path, "wb") as fout:
            try:
                torch.save(obj, fout, pickle_module=cloudpickle)
            except Exception as e:
                raise e

        self.run.save(local_path, base_path=self.run.dir)
        return local_path

    def watch(self, *args, **kwargs):
        self.run.watch(*args, **kwargs)

    def close(self):
        self.dump()

    def record(self, tag, value, global_step=None, exclude=None):
        # Let wandb figure it out.
        self.run.log({tag: value}, step=global_step, commit=False)

    def dump(self, step=None):
        self.run.log({}, step=step, commit=True)


def deploy(host: str = "", sweep_definition: Optional[str] = None, wandb_kwargs: Optional[dict] = None,
           extra_slurm_headers: Optional[List[str]] = None, extra_modules: Optional[List[str]] = None,
           debug: bool = False, interactive: bool = False, parallel_jobs: int = 1,
           tag_experiment: bool = True) -> WandbWrapper:
    """
    :param host: The host to deploy to.
    :param sweep_definition: Yaml file with the sweep configuration or sweep_id.
    :param parallel_jobs: The number of parallel jobs to run.
    :param wandb_kwargs: Kwargs to pass to wandb.init
    :param extra_slurm_headers: Extra slurm headers to add to the job script
    :param extra_modules: Extra modules to module load
    :param debug: If true does not run jobs in the cluster and invokes wandb.init with disabled=True.
    :param interactive: Not yet implemented.
    :param tag_experiment: If true creates a git tag of the current repo. Default True.
    :return: A tensorboard-like object that can be used to log data.
    """

    if wandb_kwargs is None:
        wandb_kwargs = {}
    if extra_modules is None:
        extra_modules = [
            "python/3.7",
            "libffi"
        ]
    if extra_slurm_headers is None:
        extra_slurm_headers = []
    # ensure array not in extra_slurm_headers
    if parallel_jobs > 1:
        logging.info(f"Running {parallel_jobs} as slurm job array")
        extra_slurm_headers += [f"array=0-{parallel_jobs}"]
    slurm_kwargs = {
        "extra_modules": extra_modules,
        "extra_slurm_headers": extra_slurm_headers
    }

    if not any("python" in m for m in extra_modules):
        warnings.warn("No python module found, are you sure?")
    if interactive:
        raise NotImplementedError("Not implemented yet")

    running_on_cluster = _is_running_on_cluster()
    local_run = not host and not running_on_cluster

    try:
        git_repo = git.Repo(search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        raise ValueError(f"Could not find a git repo")

    wandb_kwargs["project"] = os.path.basename(git_repo.working_dir)

    if local_run and sweep_definition:
        raise NotImplementedError(
            "Local sweeps are not supported.\n"
            f"SLURM_JOB_ID is {os.environ.get('SLURM_JOB_ID', 'KeyError')}\n"
            f"BUDDY_IS_DEPLOYED is {os.environ.get('BUDDY_IS_DEPLOYED', 'KeyError')}\n"
        )
    if (local_run or sweep_definition) and interactive:
        raise NotImplementedError("Remote debugging requires a remote host and no sweep")

    dtm = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    if running_on_cluster:
        logging.info("using wandb")
        experiment_id = f"{git_repo.head.commit.message.strip()}"
        jid = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        jid += os.environ.get("SLURM_JOB_ID", "")
        # TODO: turn into a big switch based on scheduler
        wandb_kwargs["name"] = f"{jid}_{experiment_id}"
        logger = WandbWrapper(wandb_kwargs)
    else:
        experiment_id = _ask_experiment_id(host, sweep_definition) if not debug else "DEBUG_RUN"
        logging.info(f"experiment_id: {experiment_id}")
        if local_run:
            wandb_kwargs["name"] = experiment_id + f"_{dtm}"
            return WandbWrapper(wandb_kwargs, local_tensorboard=None, debug=debug)
        else:
            # ensure we have torch
            ensure_torch_compatibility()
            # ensure we can connect
            ssh_session = _open_ssh_session(host)
            # TODO: ensure remote venv exists from python side instead of bash. If not create it.
            # get entrypoint. Warning, hydra changes directory, ensure is properly config.
            git_url = git_repo.remotes[0].url
            entrypoint = os.path.relpath(sys.argv[0], git_repo.working_dir)

            if sweep_definition:
                # TODO: change to use `python -O -u entrypoint --multirun` instead to wrap any hyperopt
                sweep_id = _load_sweep(entrypoint, sweep_definition, wandb_kwargs)
                entrypoint = f"wandb agent {sweep_id}"
            else:
                # TODO: support for distributed training with torchrun
                entrypoint = f"python -O -u {entrypoint}"
            # commit to slurm and commit to git
            scripts_folder, hash_commit = _commit(ssh_session, experiment_id, git_repo, **slurm_kwargs,
                                                  tag_experiment=tag_experiment)
            # launch jobs
            send_jobs(ssh_session, scripts_folder, git_url, hash_commit, entrypoint)
            sys.exit()

    return logger


def send_jobs(ssh_session: fabric.Connection, scripts_folder: str, git_url: str, hash_commit: str, entrypoint: str,
              proc_num: int = 1) -> None:
    ssh_command = f"bash -l {scripts_folder}/run_experiment.sh {git_url} {hash_commit} '{entrypoint}'"
    logging.info("monitor your run on https://wandb.ai/")
    logging.debug(ssh_command)
    for _ in tqdm.trange(proc_num):
        time.sleep(1)
        ssh_session.run(ssh_command)


def ensure_torch_compatibility() -> None:
    if not os.path.exists("requirements.txt"):
        return

    with open("requirements.txt") as fin:
        reqs = fin.read()
        # torch, vision or audio.
        matches = re.search(r"torch==.*cu.*", reqs)
        if "torch" in reqs and not matches:
            # https://mila-umontreal.slack.com/archives/CFAS8455H/p1624292393273100?thread_ts=1624290747.269100&cid=CFAS8455H
            warnings.warn(
                """torch rocm4.2 version will be installed on the cluster which is not supported specify torch==1.7.1+cu110 in requirements.txt instead""")


def _ask_experiment_id(host: str, sweep: str) -> str:
    title = f'{"[CLUSTER" if host else "[LOCAL"}'
    if sweep:
        title = f"{title}-SWEEP"
    title = f"{title}]"

    try:
        import tkinter.simpledialog  # fails on the server or colab
        logging.info("Name your run in the pop-up window!")
        root = tkinter.Tk()
        root.withdraw()
        experiment_id = tkinter.simpledialog.askstring(title, "experiment_id")
        root.destroy()
    except:
        if os.environ.get('BUDDY_CURRENT_TESTING_BRANCH', ''):
            import uuid
            experiment_id = f'TESTING_BRANCH-{os.environ["BUDDY_CURRENT_TESTING_BRANCH"]}-{uuid.uuid4()}'
        else:
            experiment_id = input(f"Running on {title}\ndescribe your experiment (experiment_id):\n")

    experiment_id = (experiment_id or "no_id").replace(" ", "_")
    if host:
        experiment_id = f"[CLUSTER] {experiment_id}"
    return experiment_id


def _setup_tb(logdir):
    logging.info("http://localhost:6006")
    # TODO: Use aim as local tensorboard?
    import torch.utils.tensorboard
    return torch.utils.tensorboard.SummaryWriter(log_dir=logdir)


def _open_ssh_session(hostname: str) -> fabric.Connection:
    try:
        ssh_session = fabric.Connection(host=hostname, connect_timeout=10, forward_agent=True)
        out = ssh_session.run("hostname")
        logging.debug(f"Connected to {out.stdout.strip()}")
    except SSHException as e:
        raise SSHException(
            "SSH connection failed!,"
            f"Make sure you can successfully run `ssh {hostname}` with no parameters, "
            f"any parameters should be set in the ssh_config file"
        )
    return ssh_session


def _ensure_scripts_directory(ssh_session: fabric.Connection, working_dir: str, extra_modules: List[str],
                              extra_slurm_header: List[str]) -> str:
    retr = ssh_session.run("mktemp -d -t experiment_buddy-XXXXXXXXXX")
    remote_tmp_folder = retr.stdout.strip() + "/"
    ssh_session.put(f'{SCRIPTS_PATH}/common/common.sh', remote_tmp_folder)

    scripts_dir = os.path.join(SCRIPTS_PATH, experiment_buddy.utils.get_backend(ssh_session, working_dir))

    for file in os.listdir(scripts_dir):
        fname = os.path.join(scripts_dir, file)
        if file in ("run_sweep.sh", "srun_python.sh"):
            if extra_slurm_header or extra_modules:
                fname = _insert_extra_header_and_modules(fname, extra_slurm_header, extra_modules)
                logging.debug(f"Inserted headers: {extra_slurm_header}, modules: {extra_modules}. "
                              f"File saved locally at {fname}.")
            ssh_session.put(fname, remote_tmp_folder)
        else:
            ssh_session.put(fname, remote_tmp_folder)

    return remote_tmp_folder


def _insert_extra_header_and_modules(script_path: str, extra_slurm_header: List[str],
                                     extra_slurm_modules: List[str]) -> str:
    tmp_script_path = f"/tmp/{os.path.basename(script_path)}"
    with open(script_path) as f_in, open(tmp_script_path, "w") as f_out:
        rows = f_in.readlines()
        first_free_idx = 1 + next(i for i in reversed(range(len(rows))) if "#SBATCH" in rows[i])
        header_to_insert = ""
        for h in extra_slurm_header:
            header_to_insert += f"#SBATCH --{h}\n"
        rows.insert(first_free_idx, header_to_insert)
        module_to_load = "module load " + " ".join(extra_slurm_modules) + "\n"
        rows.insert(rows.index("module purge\n") + 1, module_to_load)
        f_out.write("\n".join(rows))
    return tmp_script_path


def _check_or_copy_wandb_key(ssh_session: fabric.Connection) -> None:
    try:
        ssh_session.run("test -f $HOME/.netrc")
    except UnexpectedExit:
        logging.error(f"Wandb api key not found in {ssh_session.host}. Copying it from {DEFAULT_WANDB_KEY}")
        ssh_session.put(DEFAULT_WANDB_KEY, ".netrc")


def log_cmd(cmd, retr):
    print("################################################################")
    print(f"## {cmd}")
    print("################################################################")
    print(retr)
    print("################################################################")


def _commit(ssh_session: fabric.Connection, experiment_id: str, git_repo: git.Repo, extra_slurm_headers: List[str],
            extra_modules: List[str], tag_experiment: bool = True) -> Tuple[str, str]:
    # commit to slurm
    if experiment_id.endswith("!!"):
        extra_slurm_headers += ["partition=unkillable"]
    elif experiment_id.endswith("!"):
        extra_slurm_headers += ["partition=main"]

    scripts_folder = _commit_to_slurm(ssh_session, git_repo.working_dir, extra_modules, extra_slurm_headers)

    hash_commit = git_repo.commit().hexsha
    # commit to git
    if tag_experiment:
        hash_commit = git_sync(experiment_id, git_repo)
    return scripts_folder, hash_commit


def _commit_to_slurm(ssh_session: fabric.Connection, working_dir: str, extra_modules: List[str],
                     extra_slurm_header: List[str]) -> str:
    scripts_folder = _ensure_scripts_directory(ssh_session, working_dir, extra_modules, extra_slurm_header)
    _check_or_copy_wandb_key(ssh_session)
    return scripts_folder


def _load_sweep(entrypoint: str, sweep_definition: str, wandb_kwargs: Dict[str, str]) -> str:
    entity = wandb.Api().default_entity

    if sweep_definition.endswith(".yaml"):
        with open(sweep_definition) as f:
            data_loaded = yaml.load(f, Loader=yaml.FullLoader)
        if data_loaded["program"] != entrypoint:
            warnings.warn(f'YAML {data_loaded["program"]} does not match the entrypoint {entrypoint}')
        sweep_id = wandb.sweep(data_loaded, project=wandb_kwargs["project"], entity=entity)
    else:
        sweep_id = sweep_definition
    return os.path.join(entity, wandb_kwargs["project"], sweep_id)


def git_sync(experiment_id, repo: git.Repo) -> str:
    if any(url.lower().startswith('https://') for url in repo.remote('origin').urls):
        raise Exception("Can't use HTTPS urls for your project, please, switch to GIT urls\n"
                        "Look here for more infos https://docs.github.com/en/github/getting-started-with-github/"
                        "getting-started-with-git/managing-remote-repositories#changing-a-remote-repositorys-url")
    active_branch = repo.active_branch.name
    repo.git.checkout(detach=True)
    repo.git.add('.')
    try:
        repo.git.commit('-m', f'{experiment_id}', no_verify=True)
    except git.exc.GitCommandError:
        git_hash = repo.commit().hexsha
        repo.git.push(repo.remotes[0], active_branch)
    else:
        git_hash = repo.commit().hexsha
        tag_name = f"snapshot/{active_branch}/{git_hash}"
        repo.git.tag(tag_name)
        repo.git.push(repo.remotes[0], tag_name)
        repo.git.reset('HEAD~1')
    finally:
        repo.git.checkout(active_branch)
    return git_hash
