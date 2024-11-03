from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import replicate
import os
from fastapi.exceptions import HTTPException

app = FastAPI(title="Superteams task routes")


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_outputs: Optional[int] = 1
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5


class FineTuneRequest(BaseModel):
    instance_prompt: str
    class_prompt: Optional[str] = None
    num_class_images: Optional[int] = 100
    train_batch_size: Optional[int] = 1
    num_train_epochs: Optional[int] = 1


@app.post("/generate")
async def generate_image(request: GenerationRequest):
    try:
        output = replicate.run(
            "bytedance/sdxl-lightning-4step:5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637",
            input={
                "width": 1024,
                "height": 1024,
                "prompt": "self-portrait of a woman, lightning in the background",
                "scheduler": "K_EULER",
                "num_outputs": 1,
                "guidance_scale": 0,
                "negative_prompt": "worst quality, low quality",
                "num_inference_steps": 4,
            },
        )
        return {"status": "success", "images": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fine-tune")
async def fine_tune_model(
    images: List[UploadFile] = File(...), request: FineTuneRequest = None
):
    try:
        # Save uploaded images temporarily
        image_paths = []
        for image in images:
            file_location = f"temp/{image.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(await image.read())
            image_paths.append(file_location)

        # Start fine-tuning process
        training = replicate.training.create(
            version="stability-ai/sdxl:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5",
            input={
                "input_images": image_paths,
                "instance_prompt": request.instance_prompt,
                "class_prompt": request.class_prompt,
                "num_class_images": request.num_class_images,
                "train_batch_size": request.train_batch_size,
                "num_train_epochs": request.num_train_epochs,
            },
        )

        # Clean up temporary files
        for path in image_paths:
            os.remove(path)

        return {
            "status": "success",
            "training_id": training.id,
            "status_url": training.status_url,
        }
    except Exception as e:
        # Clean up temporary files in case of error
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))
