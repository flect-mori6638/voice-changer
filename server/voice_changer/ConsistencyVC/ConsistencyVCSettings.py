from dataclasses import dataclass, field


@dataclass
class ConsistencyVCSettings:
    gpu: int = -9999
    dstId: int = 0

    f0Detector: str = "whisper"  # whisper or ppg
    tran: int = 12
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 4

    filter_length: int = 1024
    n_mel_channels: int = 80
    sampling_rate: int = 16000
    hop_length: int = 320
    win_length: int = 1024
    mel_fmin: float = 0.0
    mel_fmax:any = None


    protect: float = 0.5
    rvcQuality: int = 0
    silenceFront: int = 1  # 0:off, 1:on
    modelSamplingRate: int = 48000

    speakers: dict[str, int] = field(default_factory=lambda: {})
    # isHalf: int = 1  # 0:off, 1:on
    # enableDirectML: int = 0  # 0:off, 1:on
    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "dstId",
        "tran",
        "extraConvertSize",
        "silenceFront",
        "filter_length",
        "n_mel_channels",
        
    ]
    floatData = ["silentThreshold", "indexRatio", "protect"]
    strData = ["f0Detector"]
