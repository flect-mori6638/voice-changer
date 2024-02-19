from torch import device

from const import EmbedderType
from voice_changer.ConsistencyVC.embedder.Embedder import Embedder
from voice_changer.ConsistencyVC.embedder.Whisper import Whisper
# from voice_changer.RVC.embedder.FairseqHubert import FairseqHubert
# from voice_changer.RVC.embedder.FairseqHubertJp import FairseqHubertJp
# from voice_changer.RVC.embedder.OnnxContentvec import OnnxContentvec
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class EmbedderManager:
    currentEmbedder: Embedder | None = None
    params: VoiceChangerParams

    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getEmbedder(
        cls, embederType: EmbedderType, isHalf: bool, dev: device
    ) -> Embedder:
        if cls.currentEmbedder is None:
            print("[Voice Changer] generate new embedder. (no embedder)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        elif cls.currentEmbedder.matchCondition(embederType) is False:
            print(embederType,":",cls.currentEmbedder.matchCondition(embederType))
            print("[Voice Changer] generate new embedder. (not match)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        else:
            print("[Voice Changer] generate new embedder. (anyway)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)

            # cls.currentEmbedder.setDevice(dev)
            # cls.currentEmbedder.setHalf(isHalf)
        return cls.currentEmbedder

    @classmethod
    def loadEmbedder(
        cls, embederType: EmbedderType, isHalf: bool, dev: device
    ) -> Embedder:
        file = cls.params.whisper_medium
        return Whisper().loadModel(file,dev)
        #PPGは未実装
        # if embederType == "whisper":
        #     file = cls.params.whisper_medium
        #     return Whisper().loadModel(file,dev)
        # else:
        #     return FairseqHubert().loadModel(file, dev, isHalf)
