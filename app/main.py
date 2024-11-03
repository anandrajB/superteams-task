import contextlib
import os
import tempfile
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

from fastapi import FastAPI, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .replicate import ReplicateApieHandler
from .schemas import FineTuneRequest, GenerationRequest
from .utils import ReplicateUtilsEnum, ResponseStatusEnum, create_zip_from_files

app = FastAPI(title="Superteams task routes")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates", extensions=["jinja2.ext.do"])
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("prompt.html", {"request": request})


@app.post(
    "/generate",
    response_model=GenerationRequest,
    summary="Generate images with replicate API",
    response_description="creates images with the user prompts and returns as blob or file object",
)
async def replicate_generate_images(request: GenerationRequest):
    with contextlib.suppress(Exception):
        try:
            os.remove("static/output.png")
        except:
            pass
        ReplicateApieHandler(request_type=ReplicateUtilsEnum.GENERATE).generate_images(
            input_prompt=request.prompt
        )

        json_compatible_item_data = jsonable_encoder(
            {"status": ResponseStatusEnum.SUCCESS, "data": "image generated"}
        )
        return JSONResponse(content=json_compatible_item_data)
    raise HTTPException(status_code=404, detail=ResponseStatusEnum.SWR.value)


@app.post(
    "/fine-tune",
    summary="Upload your images / zip files to fine tune",
    response_description="triggers the replicate fine tune trainings and response with the training id ",
)
async def fine_tune_model(
    # files: Optional[List[UploadFile]] = None,
    # zip_file: Optional[UploadFile] = None,
):
    # try:
    # if files is not None:
    #     await create_zip_from_files(files)

    training = ReplicateApieHandler(request_type=ReplicateUtilsEnum.FINETUNE).fine_tune(
        input_file="https://tfm-storage.blr1.cdn.digitaloceanspaces.com/base-training.zip"
    )
    training = None
    return {"status": "success", "training_id": training}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
