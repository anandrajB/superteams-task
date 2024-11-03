from enum import StrEnum


class ReplicateUtilsEnum(StrEnum):
    GENERATE = "GENERATE"
    FINETUNE = "FINETUNE"

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self):
        return f"{self.value}"
