import os

from data.ModelSlot import ConsistencyVCModelSlot
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.ModelSlotGenerator import ModelSlotGenerator

class ConsistencyVCModelSlotGenerator(ModelSlotGenerator):
    @classmethod
    def loadModel(cls, props: LoadModelParams):
        slotInfo: ConsistencyVCModelSlot = ConsistencyVCModelSlot()
        for file in props.files:
            if file.kind == "consistencyVCModel":
                slotInfo.modelFile = file.name
        slotInfo.name = os.path.splitext(os.path.basename(slotInfo.modelFile))[0]
        slotInfo.slotIndex = props.slot
        return slotInfo