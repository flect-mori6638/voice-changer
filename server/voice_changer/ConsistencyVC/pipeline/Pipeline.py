from typing import Any
import torch
import torch.nn.functional as F

import numpy as np

from torch.cuda.amp import autocast
from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    DeviceChangingException,
    HalfPrecisionChangingException,
    NotEnoughDataExtimateF0,
)

from mods.log_control import VoiceChangaerLogger
from voice_changer.ConsistencyVC.inferencer.Inferencer import Inferencer
from voice_changer.ConsistencyVC.embedder.Embedder import Embedder

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.VolumeExtractor import VolumeExtractor

from torchaudio.transforms import Resample

from voice_changer.utils.Timer import Timer2

class Pipeline(object):
    embedder: Embedder
    inferencer: Inferencer
    pitchExtractor: PitchExtractor

    big_npy: Any | None

    targetSR: int
    device: torch.device
    isHalf: bool

    def __init__(
        self,
        embedder: Embedder,
        inferencer: Inferencer,
        pitchExtractor: PitchExtractor,
        # feature: Any | None,
        targetSR,
        device,
        isHalf,
        resamplerIn: Resample,
        resamplerOut: Resample,
    ):
        self.embedder = embedder
        self.inferencer = inferencer
        self.pitchExtractor = pitchExtractor

        self.isHalf = isHalf
        self.device = device
        self.resamplerIn = resamplerIn
        self.resamplerOut = resamplerOut
        self.volumeExtractor = VolumeExtractor(0.5)
        self.hop_size = 320
        self.sr = 16000

    def getPipelineInfo(self):
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.getEmbedderInfo()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {"inferencer": inferencerInfo, "embedder": embedderInfo, "pitchExtractor": pitchExtractorInfo, "isHalf": self.isHalf}
    
    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor

    @torch.no_grad()
    def extract_volume_and_mask(self, audio: torch.Tensor, threshold: float):
        volume_t = self.volumeExtractor.extract_t(audio)
        mask = self.volumeExtractor.get_mask_from_volume_t(volume_t, self.inferencer_block_size, threshold=threshold)
        volume = volume_t.unsqueeze(-1).unsqueeze(0)
        return volume, mask

    def exec(
            self,
            audio
            ):
                # print("---------- pipe line --------------------")
                #print("audio",audio_t)
                #audio = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
                with Timer2("pre-process", False) as t:
                    
                    with autocast(enabled=self.isHalf):
                        try:
                            feats = self.embedder.extractFeatures(audio)
                            if torch.isnan(feats).all():
                                raise DeviceCannotSupportHalfPrecisionException()
                        except RuntimeError as e:
                            if "HALF" in e.__str__().upper():
                                raise HalfPrecisionChangingException()
                            elif "same device" in e.__str__():
                                raise DeviceChangingException()
                            else:
                                raise e
                    #feats = F.interpolate(feats.permute(0, 2, 1), size=int(n_frames), mode="nearest").permute(0, 2, 1)

                with Timer2("pre-process", False) as t:
                        #print("inferencer:",self.inferencer)
                        # 推論実行
                        audio = torch.from_numpy(audio).float().unsqueeze(0).cpu()
                        #tmp = torch.from_numpy(audio).float().unsqueeze(0)
                        audio1 = self.inferencer.infer(audio, feats)

                        #print(audio1)
                        return audio1
                        #audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

                # with Timer2("pre-process", False) as t:  # NOQA
                #     feats_buffer = feats.squeeze(0).detach().cpu()
                #     if pitch is not None:
                #         pitch_buffer = pitch.squeeze(0).detach().cpu()
                #     else:
                #         pitch_buffer = None

                #     del pitch, pitchf, feats, sid
                #     audio1 = self.resamplerOut(audio1.float())
                #     # print("[Timer::5: ]", t.secs)
                #     return audio1

                        



                        

                        
                       



