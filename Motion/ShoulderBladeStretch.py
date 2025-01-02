import numpy as np
from Tool import *
import time

class ShoulderBladeStretch:

    def __init__(self, landmarks):

        # 關節點輸入
        self.Points = [landmarks['LEFT_HIP'], landmarks['LEFT_SHOULDER'], landmarks['LEFT_WRIST']]
        
        self.Example = 'Dataset\Points\ShoulderBladeStretch\Points\ShoulderBladeStretch-1.npy'

    def process(self, isTimeOut, PointRecord):
        PointRecord.append(self.Points)      

        if isTimeOut:

            Sequences = np.load(self.Example)
            ReferenceSequence = []  
            for seq in Sequences:
                select = [seq[6], seq[0], seq[4]]
                ReferenceSequence.append(select)    
    
            return PointRecord, ReferenceSequence
              
        return PointRecord, None





    





