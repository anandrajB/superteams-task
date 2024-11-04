from enum import StrEnum


class BaseEnum(StrEnum):
    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self):
        return f"{self.value}"


class ReplicateUtilsEnum(BaseEnum):
    GENERATE = "GENERATE"
    FINETUNE = "FINETUNE"


class ResponseStatusEnum(BaseEnum):

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    SWR = "Something went wrong"
    UNAUTHORIZED = "UNAUTHORIZED"
    IGS = "Image Generated Successfully"

