from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import verify_get_request

from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import TournamentType


# TODO before submitting: replace with the pushed public repo URL and the
# full 40-char commit SHA from `git rev-parse HEAD` on that commit (branch
# names are rejected). See docs/miner.md "Submitting Your Training Repository".
IMAGE_TOURNAMENT_REPO = "https://github.com/YashShah888/REPLACE_ME"
IMAGE_TOURNAMENT_COMMIT_HASH = "0000000000000000000000000000000000000000"


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    if task_type != TournamentType.IMAGE:
        # We only enter the Image Tournament; a 404 here tells the validator
        # not to register us for text/environment tournaments instead of
        # submitting untested code (and paying an entry fee) for those.
        raise HTTPException(status_code=404, detail=f"Not participating in {task_type.value} tournaments")

    return TrainingRepoResponse(
        github_repo=IMAGE_TOURNAMENT_REPO,
        commit_hash=IMAGE_TOURNAMENT_COMMIT_HASH,
        github_token=None,
        requested_datasets=None,
    )


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/training_repo/{task_type}",
        get_training_repo,
        tags=["Subnet"],
        methods=["GET"],
        response_model=TrainingRepoResponse,
        summary="Get Training Repo",
        description="Retrieve the training repository and commit hash for the tournament.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    return router
