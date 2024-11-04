import contextlib
import os
import zipfile
from typing import List, Optional

from fastapi import APIRouter, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from .replicate import ReplicateApieHandler
from .schemas import FineTuneRequest, GenerationRequest
from .utils import ReplicateUtilsEnum, ResponseStatusEnum, create_zip_from_files

router = APIRouter(
    prefix="",
    tags=["replicate-api"],
    responses={404: {"description": "replicate api image generation and fine tune "}},
)


@router.post(
    "/generate/",
    response_model=GenerationRequest,
    summary="Generate images with replicate API",
    response_description="creates images with the user prompts and returns as blob or file object",
)
async def replicate_generate_images(request: GenerationRequest):
    with contextlib.suppress(Exception):
        ReplicateApieHandler(request_type=ReplicateUtilsEnum.GENERATE).generate_images(
            request
        )

        json_compatible_item_data = jsonable_encoder(
            {
                "status": ResponseStatusEnum.SUCCESS,
                "data": ResponseStatusEnum.IGS,
                "url": "https://superteams-task.onrender.com/generated_image/",
            }
        )
        return JSONResponse(content=json_compatible_item_data)
    raise HTTPException(status_code=404, detail=ResponseStatusEnum.SWR.value)


@router.post(
    "/fine-tune/",
    summary="Upload your images / zip files to fine tune , in the request either you can give the zip file or upload your own images to fine tune",
    response_description="triggers the replicate fine tune trainings and response with the training id ",
)
async def fine_tune_model(
    request: FineTuneRequest, files: Optional[List[UploadFile]] = None
):

    try:
        if files is not None:
            await create_zip_from_files(files)

            with zipfile.ZipFile("files.zip", "r") as zip_ref:
                file_names = zip_ref.namelist()

        training = ReplicateApieHandler(
            request_type=ReplicateUtilsEnum.FINETUNE
        ).fine_tune(request)
        return {"status": "success", "training_id": training.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        with contextlib.suppress(Exception):
            os.remove("files.zip")
