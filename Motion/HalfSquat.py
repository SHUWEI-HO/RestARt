import numpy as np
from Tool import *
import time

class HalfSquat:

    def __init__(self, landmarks):

        # 關節點輸入
        self.Points = [landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'], landmarks['LEFT_ANKLE']]
        # 載入範例動作數據
        self.Example = 'Dataset\Points\HalfSquat\Points\HalfSquat-1.npy'

     
    def process(self, isTimeOut, PointRecord):
        PointRecord.append(self.Points)      

        if isTimeOut:

            Sequences = np.load(self.Example)
            ReferenceSequence = []  
            for seq in Sequences:
                select = [seq[6], seq[8], seq[10]]
                ReferenceSequence.append(select)    
            # breakpoint()
            return PointRecord, ReferenceSequence
              
        return PointRecord, None





    





