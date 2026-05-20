import asyncio
import random
from uuid import UUID

from core import constants as cst
from core.models.payload_models import DockerEvaluationResults
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from validator.core import constants as vcst
from validator.db.database import PSQLDB
from validator.evaluation.basilica import run_basilica_eval_repos
from validator.evaluation.db_utils import load_eval_pair_state_for_models
from validator.evaluation.utils import create_basilica_eval_runner_source
from validator.evaluation.utils import normalize_rewards_and_compute_loss
from validator.evaluation.utils import process_evaluation_results
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def _db_read_with_retry(coro_factory, op_name: str):
    last_exc = None
    for attempt in range(1, vcst.EVAL_DB_RETRY_ATTEMPTS + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            delay = vcst.EVAL_DB_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, 0.3)
            if attempt < vcst.EVAL_DB_RETRY_ATTEMPTS:
                logger.warning(
                    f"DB read op '{op_name}' failed attempt {attempt}/{vcst.EVAL_DB_RETRY_ATTEMPTS}: {exc}; "
                    f"retrying in {delay + jitter:.2f}s"
                )
                await asyncio.sleep(delay + jitter)
            else:
                logger.error(f"DB read op '{op_name}' failed after {vcst.EVAL_DB_RETRY_ATTEMPTS} attempts: {exc}")
    raise last_exc


def _collect_repo_evaluation_results(models: list[str], repo_results: dict[str, dict | str]) -> dict[str, dict | str | int]:
    evaluation_results: dict[str, dict | str | int] = {}
    model_params_count = 0

    for repo in models:
        raw_result = repo_results.get(repo)
        if not isinstance(raw_result, dict):
            evaluation_results[repo] = str(raw_result)
            continue

        if raw_result.get("model_params_count") and model_params_count == 0:
            model_params_count = raw_result["model_params_count"]

        if repo in raw_result:
            evaluation_results[repo] = raw_result[repo]
            continue

        candidate_keys = [key for key in raw_result.keys() if key != "model_params_count"]
        if len(candidate_keys) == 1:
            evaluation_results[repo] = raw_result[candidate_keys[0]]
        else:
            evaluation_results[repo] = f"Evaluation failed: missing result key for repo {repo}"

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

    return evaluation_results


async def run_evaluation_basilica_text(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType | EnvironmentDatasetType,
    file_format: FileFormat,
    num_gpus: int,
    eval_seed: int | None = None,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
) -> DockerEvaluationResults:
    deployment_ids_by_repo = {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    task_type = type(dataset_type).__name__
    is_environment_eval = isinstance(dataset_type, EnvironmentDatasetType)
    basilica_image = cst.VALIDATOR_DOCKER_IMAGE_ENV if is_environment_eval else cst.VALIDATOR_DOCKER_IMAGE
    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = ["python", "-m", "validator.evaluation.eval_instruct_text"]
    elif isinstance(dataset_type, DpoDatasetType):
        command = ["python", "-m", "validator.evaluation.eval_dpo"]
    elif isinstance(dataset_type, GrpoDatasetType):
        return await run_evaluation_basilica_grpo(
            dataset, models, original_model, dataset_type, file_format, num_gpus,
            task_id=task_id,
            psql_db=psql_db,
            deployment_ids_by_repo=deployment_ids_by_repo,
        )
    elif isinstance(dataset_type, EnvironmentDatasetType):
        command = ["python", "-m", "validator.evaluation.eval_environment"]
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")
    if not is_environment_eval and not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "Basilica text eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_text for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()
    source = create_basilica_eval_runner_source(command, cst.CONTAINER_EVAL_RESULTS_PATH)

    base_env = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if is_environment_eval:
        env_name = dataset_type.environment_name
        if env_name not in cst.ENVIRONMENT_CONFIGS:
            raise ValueError(f"Environment '{env_name}' not found. Supported: {[e.value for e in cst.EnvironmentName]}")
        base_seed = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
        base_env["ENVIRONMENT_NAME"] = env_name.value
        base_env["EVAL_SEED"] = str(base_seed)
        base_env["ENV_EVAL_TEMPERATURE"] = str(vcst.ENV_EVAL_TEMPERATURE)
        base_env["ENV_SERVER_CMD"] = vcst.ENV_SERVER_CMD_DEFAULT

    logger.debug(f"Running Basilica {task_type} evaluation (per-repo deployments) for models: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        if not is_environment_eval:
            repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await run_basilica_eval_repos(
        repos=models,
        model_name=original_model,
        task_type=task_type,
        image=basilica_image,
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
    )

    evaluation_results = _collect_repo_evaluation_results(models, repo_results)
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_basilica_grpo(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: GrpoDatasetType,
    file_format: FileFormat,
    num_gpus: int,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    deployment_ids_by_repo: dict[str, str | dict[str, str]] | None = None,
) -> DockerEvaluationResults:
    deployment_ids_by_repo = deployment_ids_by_repo or {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    """
    Run GRPO evaluation on Basilica with separate deployments per repo.
    """
    command = ["python", "-m", "validator.evaluation.eval_grpo"]
    if not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "Basilica GRPO eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_grpo for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()
    source = create_basilica_eval_runner_source(command, cst.CONTAINER_EVAL_RESULTS_PATH)

    base_environment = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }

    logger.debug(f"Starting Basilica GRPO evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_environment)
        repo_env["MODELS"] = repo
        repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await run_basilica_eval_repos(
        repos=models,
        model_name=original_model,
        task_type="grpo",
        image=cst.VALIDATOR_DOCKER_IMAGE,
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
    )

    evaluation_results = _collect_repo_evaluation_results(models, repo_results)
    evaluation_results = normalize_rewards_and_compute_loss(evaluation_results)
    logger.debug(f"Grpo evaluation results post normalization: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_basilica_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    num_gpus: int,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
) -> DockerEvaluationResults:
    deployment_ids_by_repo = {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    if not test_split_url.startswith("http://") and not test_split_url.startswith("https://"):
        raise ValueError("Basilica image eval expects TEST_SPLIT_URL to be an S3/HTTP URL.")
    command = ["/app/start.sh"]
    source = create_basilica_eval_runner_source(command, cst.CONTAINER_EVAL_RESULTS_PATH)

    base_env = {
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    }

    logger.debug(f"Starting Basilica image evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        repo_env["TEST_SPLIT_URL"] = test_split_url
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await run_basilica_eval_repos(
        repos=models,
        model_name=original_model_repo,
        task_type="image",
        image="diagonalge/tuning_validator_diffusion:basilica",
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
    )

    evaluation_results = _collect_repo_evaluation_results(models, repo_results)
    return process_evaluation_results(evaluation_results, is_image=True)
