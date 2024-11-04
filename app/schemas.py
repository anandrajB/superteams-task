from typing import Optional

from pydantic import BaseModel, Field, validator


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt for image generation")
    width: Optional[int] = Field(
        default=1024, description="Width of the generated image"
    )
    height: Optional[int] = Field(
        default=1024, description="Height of the generated image"
    )
    lora_scale: Optional[float] = Field(
        default=0.6, description="Scale factor for LoRA"
    )
    num_outputs: Optional[int] = Field(default=1, description="no of outputs")
    guidance_scale: Optional[float] = Field(
        default=7.5, description="Guidance scale for generation"
    )
    apply_watermark: Optional[bool] = Field(
        default=True, description="Whether to apply a watermark"
    )
    high_noise_frac: Optional[float] = Field(
        default=0.8, description="High noise fraction"
    )
    negative_prompt: Optional[str] = Field(
        default="", description="Negative prompt for generation"
    )
    prompt_strength: Optional[float] = Field(
        default=0.8, description="Strength of the prompt"
    )
    num_inference_steps: Optional[int] = Field(
        default=50, description="Number of inference steps"
    )

    class Config:
        populate_by_name = True
        extra = "allow"
        json_schema_extra = {
            "example": {
                "prompt": "An astronaut riding a rainbow unicorn",
                "width": 1024,
                "height": 1024,
                "lora_scale": 0.6,
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "apply_watermark": True,
                "high_noise_frac": 0.8,
                "negative_prompt": "",
                "prompt_strength": 0.8,
                "num_inference_steps": 50,
            }
        }


class FineTuneRequest(BaseModel):

    file_link: Optional[str] = None
    token_string: Optional[str] = Field(default="TOK")
    caption_prefix: Optional[str] = Field(default="a photo of an ")
    max_train_steps: Optional[int] = Field(default=1000)
    is_lora: Optional[bool] = Field(default=True)
    use_face_detection_instead: Optional[bool] = Field(default=False)

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        extra = "allow"
        json_schema_extra = {
            "example": {
                "file_link": "https://tfm-storage.blr1.cdn.digitaloceanspaces.com/base-training.zip",
                "token_string": "TOK",
                "caption_prefix": "a photo of TOK",
                "max_train_steps": 1000,
                "is_lora": True,
                "use_face_detection_instead": False,
            }
        }

    @validator("file_link")
    def validate_file_link(cls, value):
        if value is not None and not value.lower().endswith(".zip"):
            raise ValueError("The file_link must be a ZIP file")
        return value
