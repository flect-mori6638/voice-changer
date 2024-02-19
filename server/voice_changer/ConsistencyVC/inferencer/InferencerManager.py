from const import ConsistencyVCInferenceTypes
from voice_changer.ConsistencyVC.inferencer.ConsistencyVCInferencer import ConsistencyVCInferencer
from voice_changer.ConsistencyVC.inferencer.Inferencer import Inferencer
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
import os


class InferencerManager:
    currentInferencer: Inferencer | None = None
    params: VoiceChangerParams
    
    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getInferencer(
        cls,
        inferencerType: ConsistencyVCInferenceTypes,
        file: str,
        gpu: int,
    ) -> Inferencer:
        cls.currentInferencer = cls.loadInferencer(inferencerType, file, gpu)
        return cls.currentInferencer

    @classmethod
    def loadInferencer(
        cls,
        inferencerType: ConsistencyVCInferenceTypes,
        file: str,
        gpu: int,
    ) -> Inferencer:   
        if inferencerType == "whisper":
            whisper_pretrain = os.path.join(os.path.dirname(cls.params.whisper_medium), "medium.pt")
            return ConsistencyVCInferencer(whisper_pretrain).loadModel(file, gpu)
        
        else:
            raise RuntimeError("[Voice Changer] Inferencer not found", inferencerType)