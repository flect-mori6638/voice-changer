import numpy as np
import torch
from const import ConsistencyVCInferenceTypes

from voice_changer.ConsistencyVC.inferencer.Inferencer import Inferencer
from voice_changer.ConsistencyVC.inferencer.models import SynthesizerTrn
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.ConsistencyVC.inferencer.models import SpeakerEncoder
import voice_changer.ConsistencyVC.inferencer.utils as utils
from voice_changer.ConsistencyVC.inferencer.mel_processing import mel_spectrogram_torch

class ConsistencyVCInferencer(Inferencer):

    def __init__(self, ptfile):
        self.ptfile = ptfile
    
    def loadModel(self, modelfile: str, gpu: int):
        self.setProps('whisper', modelfile, True, gpu)

        model = {'inter_channels': 192, 'hidden_channels': 192, 'filter_channels': 768, 'n_heads': 2, 'n_layers': 6, 'kernel_size': 3, 'p_dropout': 0.1, 'resblock': '1', 'resblock_kernel_sizes': [3, 7, 11], 'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 'upsample_rates': [10, 8, 2, 2], 'upsample_initial_channel': 512, 'upsample_kernel_sizes': [20, 16, 4, 4], 'n_layers_q': 3, 'use_spectral_norm': False, 'gin_channels': 256, 'ssl_dim': 1024, 'use_spk': False}
        # Loading model
        net_g = SynthesizerTrn(
        1024 // 2 + 1,
        8960 // 320,
        **model).cpu()
        _ = net_g.eval()
        #Loading checkpoint

        _ = utils.load_checkpoint(modelfile, net_g, None, True)
        self.net_g = net_g
        return self

    
    def infer(
        self,
        audio_t: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        
        mel_tgt = mel_spectrogram_torch(
            audio_t, 
            1024,
            80,
            16000,
            320,
            1024,
            0.0,
            None
        )
        
        c=feats
        c=c.transpose(1,0)
        c=c.unsqueeze(0)

        audio = self.net_g.infer(c.cpu(), mel=mel_tgt)
        
        return audio