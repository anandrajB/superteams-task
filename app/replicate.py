import contextlib
import os
from typing import Dict, Literal

import replicate
from pydantic.dataclasses import dataclass

from app.utils import ReplicateUtilsEnum


@dataclass
class ReplicateApieHandler:
    request_type: Literal[ReplicateUtilsEnum.GENERATE, ReplicateUtilsEnum.FINETUNE]

    def generate_images(self, input_prompt) -> None:
        with contextlib.suppress(Exception):
            os.remove("static/output.png")
        output = replicate.run(
            "anandrajb/sdxl-anand-superteams:43f9c86c5734335d2e7e80b1f83f480a687fbf459ab05739c4c4d9debdd62099",
            input={
                "width": 1024,
                "height": 1024,
                "prompt": input_prompt,
                "refine": "no_refiner",
                "scheduler": "K_EULER",
                "lora_scale": 0.6,
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "apply_watermark": False,
                "high_noise_frac": 0.8,
                "negative_prompt": "",
                "prompt_strength": 0.8,
                "num_inference_steps": 50,
            },
        )
        with open("static/output.png", "wb") as f:
            f.write(output[0].read())
        for file_output in output:
            with open("static/output.png", "wb") as f:
                f.write(file_output.read())
        return None

    def fine_tune(self, input_file) -> Dict:
        return replicate.trainings.create(
            model="stability-ai/sdxl",
            version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "input_images": input_file,
                "token_string": "TOK",
                "caption_prefix": "a photo of TOK",
                "max_train_steps": 1000,
                "is_lora": True,
                "use_face_detection_instead": False,
            },
            destination="anandrajb/sdxl-anand-superteams",
        )
