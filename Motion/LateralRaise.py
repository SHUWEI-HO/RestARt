import numpy as np
from Tool import *
import time

class LateralRaise:
    """
    LateralRaise 類別用於處理使用者的側舉動作。
    藉由比對使用者的動作與預設範例動作數據，檢測動作是否正確。
    """

    def __init__(self, landmarks):
        """
        初始化 LateralRaise 類別，設定關節點與範例動作路徑。
        
        參數:
            landmarks (dict): Mediapipe 偵測出的關節點座標，格式為 {關節名稱: [x, y]}
        """
        # 提取使用者左側髖關節、肩膀與手腕的座標點
        self.Points = [landmarks['LEFT_HIP'], landmarks['LEFT_SHOULDER'], landmarks['LEFT_WRIST']]
        
        # 載入範例側舉動作數據檔案路徑
        self.Example = 'Dataset\\Points\\KneeRaise\\Points\\KneeRaise-1.npy'  # 注意：此路徑應修改為 LateralRaise 的範例檔案

    def process(self, isTimeOut, PointRecord):
        """
        處理使用者動作，進行動作紀錄與範例比對。
        
        參數:
            isTimeOut (bool): 是否達到超時（檢測結束）的條件
            PointRecord (list): 用於儲存使用者動作的關節點紀錄
        
        回傳:
            PointRecord (list): 更新後的使用者動作關節點紀錄
            ReferenceSequence (list or None): 若 isTimeOut 為 True，返回範例動作序列；否則返回 None
        """
        # 將當前使用者的關節點紀錄加入 PointRecord
        PointRecord.append(self.Points)      

        if isTimeOut:
            # 若檢測已達超時條件，則讀取範例動作數據進行比對
            Sequences = np.load(self.Example)  # 載入範例側舉動作數據 (numpy array)

            # 將範例動作數據轉為參考序列 (ReferenceSequence)
            ReferenceSequence = []  
            for seq in Sequences:
                # 提取範例中的左髖關節、左肩膀與左手腕作為參考點
                select = [seq[6], seq[0], seq[4]]
                ReferenceSequence.append(select)    
    
            return PointRecord, ReferenceSequence  # 回傳使用者紀錄與範例序列
              
        return PointRecord, None  # 若未達超時條件，僅回傳使用者紀錄
