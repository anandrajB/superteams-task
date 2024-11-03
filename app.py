from pydantic.dataclasses import dataclass
from typing import Literal
from utils import ReplicateUtilsEnum
import replicate
import os


@dataclass
class ReplicateImageHandler:
    request_type: Literal[ReplicateUtilsEnum.GENERATE, ReplicateUtilsEnum.FINETUNE]

    def generate_images(self, input_prompt) -> None:
        output = replicate.run(
            os.environ.get("REPLICATE_IMAGE_GENERATION_MODEL"),
            input={
                "width": 1024,
                "height": 1024,
                "prompt": input_prompt,
                "scheduler": "K_EULER",
                "num_outputs": 1,
                "guidance_scale": 0,
                "negative_prompt": "worst quality, low quality",
                "num_inference_steps": 4,
            },
        )
        with open("output.png", "wb") as f:
            f.write(output[0].read())
        for idx, file_output in enumerate(output):
            with open(f"output_{idx}.png", "wb") as f:
                f.write(file_output.read())
        return None

    def fine_tune(self):
        return None
