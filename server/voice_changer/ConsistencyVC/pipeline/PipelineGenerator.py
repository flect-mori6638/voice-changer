import traceback
from Exceptions import PipelineCreateException
from data.ModelSlot import ConsistencyVCModelSlot

from voice_changer.ConsistencyVC.inferencer.InferencerManager import InferencerManager
from voice_changer.ConsistencyVC.pipeline.Pipeline import Pipeline
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager


from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.ConsistencyVC.embedder.EmbedderManager import EmbedderManager
#from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
import os
import torch

from torchaudio.transforms import Resample

from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager

def createPipeline(modelSlot: ConsistencyVCModelSlot, gpu: int, f0Detector: str, inputSampleRate: int, outputSampleRate: int):
    dev = DeviceManager.get_instance().getDevice(gpu)
    vcparams = VoiceChangerParamsManager.get_instance().params
    # half = DeviceManager.get_instance().halfPrecisionAvailable(gpu)
    half = False

    # Inferencer 生成
    try:        
        modelPath = os.path.join(vcparams.model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.modelFile))
        inferencer = InferencerManager.getInferencer(modelSlot.modelType, modelPath, gpu)
        print("modelSlot.modelType",modelSlot.modelType)
        print("getInferencer",inferencer)
    except Exception as e:
        print("[Voice Changer] exception! loading inferencer", e)
        traceback.print_exc()
        raise PipelineCreateException("[Voice Changer] exception! loading inferencer")

    # Embedder 生成
    try:
        embedder = EmbedderManager.getEmbedder(
            modelSlot.embedder,
            # emmbedderFilename,
            half,
            dev,
        )
    except Exception as e:
        print("[Voice Changer]  exception! loading embedder", e)
        traceback.print_exc()
        raise PipelineCreateException("[Voice Changer] exception! loading embedder")

    # pitchExtractor
    pitchExtractor = PitchExtractorManager.getPitchExtractor(f0Detector, gpu)

    resamplerIn = Resample(inputSampleRate, 16000, dtype=torch.int16).to(dev)
    resamplerOut = Resample(16000, outputSampleRate, dtype=torch.int16).to(dev)

    pipeline = Pipeline(
        embedder,
        inferencer,
        pitchExtractor,
        16000,
        dev,
        half,
        resamplerIn,
        resamplerOut
    )

    return pipeline
