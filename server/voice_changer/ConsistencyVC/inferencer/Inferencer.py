from typing import Any, Protocol
import torch
import onnxruntime

from const import ConsistencyVCInferenceTypes
#from models import SpeakerEncoder

class Inferencer(Protocol):
    inferencerType: ConsistencyVCInferenceTypes = "Whisper"
    file: str
    isHalf: bool = True
    gpu: int = 0

    model: onnxruntime.InferenceSession | Any | None = None

    def loadModel(self, file: str, gpu: int):
        ...

    def infer(
        self,
        audio_t: torch.Tensor,
        feats: torch.Tensor
    ) -> torch.Tensor:
        ...

    def setProps(
    self,
    inferencerType: ConsistencyVCInferenceTypes,
    file: str,
    isHalf: bool,
    gpu: int,
    ):
        self.inferencerType = inferencerType
        self.file = file
        self.isHalf = isHalf
        self.gpu = gpu

    def getInferencerInfo(self):
        return {
            "inferencerType": self.inferencerType,
            "file": self.file,
            "isHalf": self.isHalf,
            "gpu": self.gpu,
        }



