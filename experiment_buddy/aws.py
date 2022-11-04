import dataclasses
import logging
import re
from typing import List, Dict

import requests.utils


@dataclasses.dataclass
class JobSpec:
    docker_tag: str
    cmd: List[str]
    job_requirements: List[Dict[str, str]]
    compute_environment: str


def _maybe_create_compute_queue(client, project_name, compute_env_name):
    queue_name = f"{project_name}-queue"
    queues = client.describe_job_queues()["jobQueues"]
    queue_names = [queue["jobQueueName"] for queue in queues]
    if queue_name not in queue_names:
        response = client.create_job_queue(
            jobQueueName=queue_name,
            state='enabled',
            priority=1,
            computeEnvironmentOrder=[
                {
                    'order': 1,
                    'computeEnvironment': compute_env_name
                },
            ],
        )
        logging.info(response)
        assert response["HTTPStatusCode"] == 200
        return response["jobQueueName"]
    return queue_name


def _maybe_create_compute_env(client, compute_env_name):
    compute_envs = client.describe_compute_environments()["computeEnvironments"]
    names = [compute_env["computeEnvironmentName"] for compute_env in compute_envs]
    if compute_env_name not in names:
        raise NotImplementedError("Compute environment not found. Please create it manually.")
    return compute_env_name


def _create_job_definition(client, project_name, docker_tag):
    response = client.register_job_definition(jobDefinitionName=project_name,
                                              type="container",
                                              containerProperties={"image": docker_tag,
                                                                   "vcpus": 8,
                                                                   "memory": 2048,
                                                                   "command": ["/bin/bash"]})
    return response["jobDefinitionName"]


def _create_job(client, project_name, experiment_id, cmd, resource_requirements: List[Dict[str, str]]):
    wandb_key = requests.utils.get_netrc_auth("https://api.wandb.ai")[-1]
    default_container = {"command": cmd, "environment": [
        {"name": "WANDB_API_KEY", "value": wandb_key},
    ], "resourceRequirements": resource_requirements}

    response = client.submit_job(jobName=experiment_id,
                                 jobQueue=f"{project_name}-queue",
                                 jobDefinition=project_name,
                                 containerOverrides=default_container,
                                 retryStrategy={"attempts": 1}, timeout={"attemptDurationSeconds": 60 * 60})
    logging.info(response)
    return response


def deploy_aws(experiment_id: str, job_spec: JobSpec, deregister: bool = True):
    try:
        import boto3
    except ImportError:
        raise ImportError("Please install boto3 to use AWS deployment")
    user, project_name, version = re.findall(r"[a-z0-9]+(?:[._-]{1,2}[a-z0-9]+)*", job_spec.docker_tag)
    client = boto3.client("batch")

    _maybe_create_compute_env(client, job_spec.compute_environment)
    _maybe_create_compute_queue(client, project_name, job_spec.compute_environment)
    _create_job_definition(client, project_name, job_spec.docker_tag)
    job_response = _create_job(client, project_name, experiment_id, job_spec.cmd, job_spec.job_requirements)

    if deregister:
        response = client.deregister_job_definition(jobDefinition=project_name)
        logging.info(response)
    return job_response
