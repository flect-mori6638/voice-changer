import os
import numpy as np

from glob import glob
from tqdm import tqdm
from  voice_changer.ConsistencyVC.embedder.whisper.model import Whisper as _Whisper, ModelDimensions
#from whisper.model import Whisper, ModelDimensions
from voice_changer.ConsistencyVC.embedder.whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
import librosa
import soundfile as sf

import torch
from torch import device
from voice_changer.ConsistencyVC.embedder.Embedder import Embedder


class Whisper(Embedder):
    def loadModel(self, file: str, dev: device, isHalf: bool = True)-> Embedder:
        super().setProps("whisper", file, dev, isHalf)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(file, map_location=dev)
        dims = ModelDimensions(**checkpoint["dims"])
        model = _Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.pretrain_model = model.to(self.device)
        return self
    
    def extractFeatures(self, audio)-> torch.Tensor:
        #if len(audio) >= sr * 29:
        #print(wavPath,"cut to 29s")
        #audio = audio[:sr * 29]
        #librosa.output.write_wav("your_audio_file.wav", audio, sr)
        #sf.write(wavPath, audio, sr)
        #audio = np.frombuffer(audio, np.int16).flatten().astype(np.float32) / 32768.0
        audln = audio.shape[0]
        ppgln = audln // 320
        mel = log_mel_spectrogram(audio).to(self.device)
        with torch.no_grad():
            ppg = self.pretrain_model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            
            if ppgln>ppg.shape[0]:
                print("ppgln>ppg.shape[0]")
            ppg = ppg[:ppgln,] # [length, dim=1024]
            #if audln // 320<ppg.shape[0]:
            #    print("audln // 320<ppg.shape[0]")
            return torch.from_numpy(ppg)
        
    # def extractFeatures(
    #     self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    # ) -> torch.Tensor:
    #     padding_mask = torch.BoolTensor(feats.shape).to(self.dev).fill_(False)

    #     # オリジナル_v1は L9にfinal_projをかけていた。(-> 256)
    #     # オリジナル_v2は L12にfinal_projをかけない。(-> 768)

    #     inputs = {
    #         "source": feats.to(self.dev),
    #         "padding_mask": padding_mask,
    #         "output_layer": embOutputLayer,  # 9 or 12
    #     }

    #     with torch.no_grad():
    #         logits = self.model.extract_features(**inputs)
    #         if useFinalProj:
    #             feats = self.model.final_proj(logits[0])
    #         else:
    #             feats = logits[0]
    #     return feats
    
    # if __name__ == "__main__":
    # # 读取所有 .wav 文件
    # data_dir = r"./dataset/"
    # wav_files = glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    # wav_files=sorted(wav_files)
    # whisper = load_model(os.path.join("whisper_pretrain", "medium.pt"))

    # for wav in tqdm(wav_files):
    #     ppg_path=wav.replace(r".wav",r"whisper.pt.npy")
    #     #print(wav,ppg_path)
    #     if not os.path.exists(ppg_path):
    #         pred_ppg(whisper, wav, ppg_path)

