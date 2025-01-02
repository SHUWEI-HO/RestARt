import numpy as np
from Tool import *
import time

class LungeStretch:

    def __init__(self, landmarks, endtime):

        # 關節點輸入
        self.Points = [landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'], landmarks['LEFT_ANKLE']]
        
        self.Example = ''
        # 時間限制
        self.endtime = endtime
     
    def process(self, StartTime):
        
        if StartTime is None:
            StartTime = time.time()
            self.PointRecord = []

        elapsedtime = time.time()-StartTime
                
        if elapsedtime > self.endtime:
            self.message = '動作完成!將進行比對' 
              
            return self.PointRecord, self.message, StartTime, np.load(self.Example)
                     
        else:
            self.message = f'保持動作，剩餘{self.endtime-elapsedtime}秒'
            self.PointRecord.append(self.Points)
        
        return self.PointRecord, self.message, StartTime, None





    





