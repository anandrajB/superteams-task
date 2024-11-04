import contextlib
import os
from typing import Dict, Literal

import replicate
from pydantic.dataclasses import dataclass

from app.enum import ReplicateUtilsEnum
from .schemas import GenerationRequest, FineTuneRequest


@dataclass
class ReplicateApieHandler:
    request_type: Literal[ReplicateUtilsEnum.GENERATE, ReplicateUtilsEnum.FINETUNE]

    def generate_images(self, request: GenerationRequest) -> None:
        with contextlib.suppress(Exception):
            os.remove("static/output.png")
        output = replicate.run(
            "anandrajb/sdxl-anand-superteams:43f9c86c5734335d2e7e80b1f83f480a687fbf459ab05739c4c4d9debdd62099",
            input={
                "width": request.width,
                "height": request.height,
                "prompt": request.prompt,
                "refine": "no_refiner",
                "scheduler": "K_EULER",
                "lora_scale": request.lora_scale,
                "num_outputs": request.num_outputs,
                "guidance_scale": request.guidance_scale,
                "apply_watermark": request.apply_watermark,
                "high_noise_frac": request.high_noise_frac,
                "negative_prompt": "",
                "prompt_strength": request.prompt_strength,
                "num_inference_steps": request.num_inference_steps,
            },
        )
        with open("static/output.png", "wb") as f:
            f.write(output[0].read())
        for file_output in output:
            with open("static/output.png", "wb") as f:
                f.write(file_output.read())
        return None

    def fine_tune(self, request: FineTuneRequest) -> Dict:
        return replicate.trainings.create(
            model="stability-ai/sdxl",
            version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "input_images": request.file_link,
                "token_string": request.token_string,
                "caption_prefix": request.caption_prefix,
                "max_train_steps": request.max_train_steps,
                "is_lora": request.is_lora,
                "use_face_detection_instead": request.use_face_detection_instead,
            },
            destination="anandrajb/sdxl-anand-superteams",
        )
