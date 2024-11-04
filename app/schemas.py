from typing import Optional

from pydantic import BaseModel


class GenerationRequest(BaseModel):
    prompt: str


class FineTuneRequest(BaseModel):
    instance_prompt: str
    class_prompt: Optional[str] = None
    num_class_images: Optional[int] = 100
    train_batch_size: Optional[int] = 1
    num_train_epochs: Optional[int] = 1
