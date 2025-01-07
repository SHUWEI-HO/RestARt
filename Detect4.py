'''
以下是會用到的套件，這邊呢不一定是python官方有的套件，
有可能是他人設定的，又或者是自己寫的類別或函式。
這邊套件的呼叫主要看你這個檔案的相對位置。什麼是這個檔案的相對位置呢?
就是依據現在這個檔案所在的路徑視為根目錄，然後依據這個往前推或往後推，
比如下面的檔案位置:
-----Detect4.py 
-----Train.py
----Photo.py
------/test/Demo.py

如果要執行Detect4.py且要呼叫Train.py中的套件:

直接呼叫 

from Train import (自定義的套件)

如果要呼叫 Photo.py 中套件的話

則
from ..Photo import (自定義的套件)

如果要呼叫 Demo.py 中套件的話

則 
from test.Demo import (自定義的套件)

以上式基礎的呼叫方法，這樣的方法可以更好的撰寫程式，寫起來絲路會越來越清晰。
如果要查看套件的話，藥用ctrl+滑鼠左鍵就可以看套件的呼叫位置

'''

import cv2
import mediapipe as mp
import argparse
import numpy as np
from Camera import check_camera
from Tool import initial_model, compute_distance, get_landmarks
from PIL import ImageFont, ImageDraw, Image
from sklearn.preprocessing import LabelEncoder
import joblib
from Motion import HalfSquat, KneeRaise, LateralRaise, ShoulderBladeStretch, LungeStretch, Pendulum
import warnings
from Contrast import contrasting
import time
import os
import socket
from cvzone.PoseModule import PoseDetector

# 忽略特定警告訊息
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------
# 解析命令行參數的函式
# --------------------------------------------------------
def get_parser():
    '''
    該函式用於解析命令行參數，方便執行時指定不同的攝像頭編號和動作模式。
    - --camera: 選擇攝像頭編號，預設為 0（主攝像頭）。
    - --mode: 選擇復健動作模式，預設為 'HalfSquat'。
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default=0, type=int)         # 攝像頭參數，預設為 0
    parser.add_argument('--mode', type=str, default='HalfSquat') # 復健模式參數，預設為 'HalfSquat'
    return parser.parse_args()

# --------------------------------------------------------
# 負責傳送動作數據與錯誤訊息的類別
# --------------------------------------------------------
class MotionErrorSender:
    '''
    該類別負責使用 socket 傳送動作數據與錯誤訊息至 Unity。
    - 包含三個 socket：
        1. motionSocket: 傳送動作數據，端口為 6720
        2. errorSocket: 傳送錯誤訊息，端口為 7801
        3. timeSocket: 傳送時間資訊，端口為 8106
    '''
    def __init__(self):
        # (1) 初始化動作數據傳輸的 socket，端口為 6720
        self.motionSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.motionAddressPort = ("127.0.0.1", 6720)

        # (2) 初始化錯誤訊息傳輸的 socket，端口為 7801
        self.errorSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.errorAddressPort = ("127.0.0.1", 7801)

        # (3) 初始化時間訊息傳輸的 socket，端口為 8106
        self.timeSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.timeAddressPort = ("127.0.0.1", 8106)

    def send_motion_data(self, data):
        '''
        傳送骨架動作數據。
        - data: 骨架動作數據的字串形式。
        '''
        self.motionSocket.sendto(data.encode(), self.motionAddressPort)

    def send_error_message(self, message):
        '''
        傳送錯誤或提示訊息。
        - message: 錯誤或提示訊息的字串形式。
        '''
        self.errorSocket.sendto(message.encode(), self.errorAddressPort)
    
    def send_time(self, time):
        '''
        傳送剩餘時間。
        - time: 剩餘時間的字串形式。
        '''
        self.timeSocket.sendto(time.encode(), self.timeAddressPort)

# --------------------------------------------------------
# 負責檢測動作與處理邏輯的主要類別
# --------------------------------------------------------
class BaseDetect:
    '''
    該類別負責動作檢測、動作模式切換、結果顯示與資料傳輸。
    '''
    def __init__(self, args):
        # 1) 保存初始參數（攝像頭編號與模式）
        self.Camera = args.camera
        self.mode = args.mode

        # 2) 建立接收 Unity 動作切換指令的 socket，端口為 5053
        self.switchSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.switchSocket.bind(("0.0.0.0", 5053))  # 監聽 5053 端口
        self.switchSocket.setblocking(False)        # 設定為非阻塞模式

        # 3) 載入初始模型，依據當前模式（mode）
        self.load_model_scaler_label(self.mode)

        # 4) 初始化 Mediapipe Pose 模組，用於偵測人體骨架
        # self.model, self.mp_pose = initial_model('Pose')

        # 5) 設定繪製字型的路徑與大小（根據實際字型路徑修改）
        self.fontpath = 'Font/msjhbd.ttc'           # 字型路徑（需更改為實際存在的路徑）
        self.font = ImageFont.truetype(self.fontpath, 32)

        # 6) 用於動作預測與骨架傳輸的緩衝區
        self.sequence_buffer = []  # 動作模型預測序列
        self.posList = []          # 用於平滑處理並傳輸至 Unity 的骨架資料

        # 7) 狀態參數初始化
        self.counter = 0     # 計算成功完成動作的次數
        self.message = None  # 訊息提示（如：保持動作）
        self.status = None   # 狀態（用於訊息顯示的顏色）
        self.EndTime = 7     # 每次檢測的時間上限（秒）
        self.label = '未檢測'   # 動作類別標籤，透過 KNN 模型預測得出
        self.confidence = 0.0  # 預測的信心指數

        # 8) 用於動作比對與錯誤訊息傳輸
        self.PointsRecord = []            # 儲存檢測過程中的人體動作數據
        self.sender = MotionErrorSender() # 初始化傳輸類別

    # --------------------------------------------------------
    # 載入或切換模型、標準化工具與編碼器
    # --------------------------------------------------------
    def load_model_scaler_label(self, mode_name):
        '''
        根據指定的模式名稱載入對應的模型、標準化工具與標籤編碼器。
        - mode_name: 要載入的動作模式名稱。
        '''
        ModelChoose = {
            'HalfSquat': 'Model/HalfSquat.pkl',
            'KneeRaise': 'Model/KneeRaise.pkl',
            'LateralRaise': 'Model/LateralRaise.pkl',
            'ShoulderBladeStretch': 'Model/ShoulderBladeStretch.pkl'
        }

        ScalerChoose = {
            'HalfSquat': 'Scaler/HalfSquatScaler.pkl',
            'KneeRaise': 'Scaler/KneeRaiseScaler.pkl',
            'LateralRaise': 'Scaler/LateralRaiseScaler.pkl',
            'ShoulderBladeStretch': 'Scaler/ShoulderBladeStretchScaler.pkl'
        }

        LabelEncChoose = {
            'HalfSquat': 'Scaler/HalfSquatLabelEncoder.pkl',
            'KneeRaise': 'Scaler/KneeRaiseLabelEncoder.pkl',
            'LateralRaise': 'Scaler/LateralRaiseLabelEncoder.pkl',
            'ShoulderBladeStretch': 'Scaler/ShoulderBladeStretchLabelEncoder.pkl'
        }

        SeqInfoChoose = {
            'HalfSquat': 'Scaler/HalfSquatSeqInfo.pkl',
            'KneeRaise': 'Scaler/KneeRaiseSeqInfo.pkl',
            'LateralRaise': 'Scaler/LateralRaiseSeqInfo.pkl',
            'ShoulderBladeStretch': 'Scaler/ShoulderBladeStretchSeqInfo.pkl'
        }

        # 確認模式名稱是否有效
        if mode_name not in ModelChoose:
            raise ValueError(f"未知的模式: {mode_name}")

        # 載入對應的模型與工具
        self.mode = mode_name
        self.Model = joblib.load(ModelChoose[mode_name])         # 載入動作辨識模型
        self.Scaler = joblib.load(ScalerChoose[mode_name])       # 載入標準化工具
        self.LabelEncoder = joblib.load(LabelEncChoose[mode_name]) # 載入標籤編碼器
        self.max_length, self.feature_dim = joblib.load(SeqInfoChoose[mode_name]) # 載入序列資訊

        print(f"[Python] 已切換 / 載入模式: {mode_name}")

    # --------------------------------------------------------
    # 嘗試接收來自 Unity 的動作切換指令
    # --------------------------------------------------------
    def try_receive_mode_switch(self, start):
        '''
        在每幀開始時嘗試接收 Unity 傳來的動作切換指令。
        - start: 當前的啟動模式。
        - 回傳: (是否切換, 新的啟動模式)
        '''
        ACTION_MAP = {
            '2': 'HalfSquat',
            '3': 'KneeRaise',
            '5': 'ShoulderBladeStretch',
            '6': 'LateralRaise',
            '8': 'Close'
        }
        try:
            # 接收資料並解碼
            data, addr = self.switchSocket.recvfrom(1024)
            message = data.decode('utf-8').strip()

            # 解析動作編號與啟動模式
            message = message.split(',')
            action_number = message[0].strip()
            start_mode = message[1].strip()
            print(f"[Python] 收到動作編號: {action_number}")
            print(f"[Python] 收到啟動信息: {start_mode}")

            # 若為已知動作編號，則切換模式
            if action_number in ACTION_MAP:
                new_mode = ACTION_MAP[action_number]
                if action_number != '8':
                    self.load_model_scaler_label(new_mode)
                print(start_mode)
                return True, start_mode
            else:
                print(f"[Python] 未知的動作編號: {action_number}")
        except BlockingIOError:
            pass  # 若未接收到資料則跳過

        return None, start

    # --------------------------------------------------------
    # 在畫面上繪製相關資訊（模式、預測、訊息、進度條）
    # --------------------------------------------------------
    def draw_info(self, image, mode, RestTime=None):
        '''
        在畫面上繪製模式、預測結果、提示訊息與進度條。
        - image: 目前的影像。
        - mode: 當前的動作模式。
        - RestTime: 剩餘時間，若為 None 則顯示滿條。
        '''
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        ModeChoose = {
            'HalfSquat': '半蹲',
            'KneeRaise': '提膝',
            'LungeStretch': '弓箭步',
            'LateralRaise': '手臂側舉',
            'ShoulderBladeStretch': '肩胛拉伸',
            'Pendulum': '鐘擺運動'
        }

        # 顯示模式 / 預測結果
        draw.rectangle([20, 390, 250, 460], fill=(255, 0, 0))
        draw.rectangle([30, 400, 240, 450], fill=(0, 255, 0))
        draw.text((35, 400), f"模式: {ModeChoose.get(mode, mode)}", font=self.font, fill=(128, 124, 0))

        # 顯示錯誤 / 提示訊息
        if self.message is not None:
            color = (120, 255, 0) if self.status else (255, 25, 0)
            draw.text((30, 70), f"訊息: {self.message}", font=self.font, fill=color)

        # 進度條(顯示剩餘時間)
        image_width, image_height = pil_image.size
        total_height = 300
        bar_width = 20
        left_x = image_width - 30 - bar_width
        top_y = (image_height - total_height) // 2
        # 背景灰色
        draw.rectangle([left_x, top_y, left_x + bar_width, top_y + total_height], fill=(200, 200, 200))

        if RestTime is not None:
            progress = RestTime / self.EndTime
            current_height = int(total_height * progress)
            draw.rectangle([left_x, top_y + (total_height - current_height),
                            left_x + bar_width, top_y + total_height], fill=(255, 165, 0))
        else:
            draw.rectangle([left_x, top_y, left_x + bar_width, top_y + total_height], fill=(255, 165, 0))

        # 更新回 OpenCV 圖片
        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # --------------------------------------------------------
    # 將累積的動作序列進行補齊或截斷，轉成 (1, feature_dim * max_length)
    # --------------------------------------------------------
    def pad_and_flatten(self, seq):
        '''
        將累積的動作序列進行補齊或截斷，並展平成一維向量。
        - seq: 當前的動作序列。
        '''
        current_length = seq.shape[0]
        if current_length < self.max_length:
            pad_length = self.max_length - current_length
            seq = np.vstack([seq, np.zeros((pad_length, self.feature_dim))])
        elif current_length > self.max_length:
            seq = seq[-self.max_length:, :]

        flattened = seq.reshape(-1)
        return flattened.reshape(1, -1)

    # --------------------------------------------------------
    # 核心迴圈: 讀取攝影機、偵測骨架、傳送資料與錯誤訊息
    # --------------------------------------------------------
    def process(self):
        cap = check_camera(self.Camera)  # 檢查攝像頭是否可用
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)  # 設置影像的寬
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)  # 設置影像的高
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 設置儲存的格式
        out = cv2.VideoWriter('output_video_.mp4', fourcc, 24, size)  # 設定儲存檔案名稱
        Detect = True  # 控制是否啟用骨架檢測
        StartTime = None  # 動作計時起始時間
        TimeOut = False  # 是否已超過動作時間
        RestTime = None  # 剩餘時間
        detector = PoseDetector()  # 初始化 Mediapipe Pose 偵測器

        ReferenceSequences = None  # 基準骨架序列，用於後續動作比對
        start_mode = 'Stop'  # 控制程式開始或停止的模式

        while True:
            # 嘗試接收是否有新的模式切換（啟動或停止）
            check, start_mode = self.try_receive_mode_switch(start_mode)

            if check:  # 若檢測到模式有變化則更新 Detect 狀態
                Detect = check
            ret, frame = cap.read()
            if not ret:  # 若無法讀取影像則結束迴圈
                break

            # 使用 cvZone 的 PoseDetector 進行骨架偵測
            img, results = detector.findPose(frame)
            lmlist, bboxinfo = detector.findPosition(img)

            # 若偵測到有人體出現在畫面中
            if bboxinfo:
                landmarks = results.pose_landmarks.landmark
                # 建立骨架點位字串 lmString
                lmString = ''
                for lm in lmlist:
                    lmString += f'{lm[0]},{img.shape[0]-lm[1]},{lm[2]},'
                self.posList.append(lmString)

                # 每 5 幀進行一次平滑化處理與資料傳送
                if len(self.posList) >= 5:
                    avgPos = []  # 儲存平滑後的骨架點
                    num_lm = 33  # Mediapipe Pose 預設 33 個關節點
                    for i in range(num_lm):
                        xPos, yPos, zPos = [], [], []  # 儲存該點的 X, Y, Z 座標
                        for j in range(len(self.posList) - 2, len(self.posList) + 3):
                            if 0 <= j < len(self.posList):
                                coords = self.posList[j].split(',')
                                if i * 3 + 2 < len(coords):
                                    xPos.append(float(coords[i * 3]))
                                    yPos.append(float(coords[i * 3 + 1]))
                                    zPos.append(float(coords[i * 3 + 2]))
                        avgX = sum(xPos) / len(xPos)
                        avgY = sum(yPos) / len(yPos)
                        avgZ = sum(zPos) / len(zPos)
                        avgPos.append(f'{avgX},{avgY},{avgZ},')  # 平均值作為該點的平滑結果
                    avgPosString = ''.join(avgPos)
                    self.sender.send_motion_data(avgPosString)  # 傳送平滑後的骨架資料

            # 如果目前模式為啟動狀態，則進行動作偵測與處理
            if start_mode == 'Going':

                if Detect:  # 若啟用偵測模式
                    if bboxinfo:  # 偵測到人體骨架
                        if StartTime is None:  # 第一次偵測到人體，開始計時
                            StartTime = time.time()
                            self.PointsRecord = []

                        # 計算已經過時間
                        ElapsedTime = time.time() - StartTime
                        if ElapsedTime >= self.EndTime:  # 若時間超過設定時間則動作完成
                            RestTime = None
                            TimeOut = True
                            self.status = True
                            self.message = '動作完成! 即將進行比對...'

                            # 在畫面上顯示提示訊息
                            ShowMessageDuration = 5.0  # 提示訊息顯示時間
                            ShowMessageStart = time.time()

                            while time.time() - ShowMessageStart < ShowMessageDuration:
                                ret_msg, frame_msg = cap.read()
                                if not ret_msg:
                                    break
                                tempFrame = frame_msg.copy()
                                if self.message is not None:
                                    self.sender.send_error_message(self.message)
                                self.draw_info(tempFrame, self.mode, None)
                                out.write(tempFrame)
                                cv2.imshow('畫面顯示', tempFrame)
                                if cv2.waitKey(10) & 0xFF == ord('q'):
                                    break
                        else:  # 若時間未超過設定時間則繼續顯示剩餘時間
                            RestTime = int(self.EndTime - ElapsedTime)
                            self.sender.send_time(str(RestTime))
                            self.message = '請保持動作~'

                        # 根據模式進行不同動作處理
                        landmarks_coords = get_landmarks(self.mp_pose, landmarks)
                        if self.mode == 'HalfSquat':
                            self.PointsRecord, ReferenceSequences = HalfSquat.HalfSquat(landmarks_coords).process(TimeOut, self.PointsRecord)
                        elif self.mode == 'KneeRaise':
                            self.PointsRecord, ReferenceSequences = KneeRaise.KneeRaise(landmarks_coords).process(TimeOut, self.PointsRecord)
                        elif self.mode == 'LateralRaise':
                            self.PointsRecord, ReferenceSequences = LateralRaise.LateralRaise(landmarks_coords).process(TimeOut, self.PointsRecord)
                        elif self.mode == 'ShoulderBladeStretch':
                            self.PointsRecord, ReferenceSequences = ShoulderBladeStretch.ShoulderBladeStretch(landmarks_coords).process(TimeOut, self.PointsRecord)

                    else:  # 未檢測到人體時的處理
                        StartTime = None
                        RestTime = None
                        TimeOut = False
                        self.message = '未檢測到人體姿勢'
                        self.status = False

                    self.draw_info(img, self.mode, RestTime)

                    # 動作完成且超時時進入對比階段
                    if TimeOut and ReferenceSequences is not None:
                        Detect = False

                else:  # Detect = False，進入比對階段
                    self.message, self.status = contrasting(self.PointsRecord, ReferenceSequences, self.mode)
                    self.sender.send_error_message(self.message)  # 傳送比對結果

                    ShowResultDuration = 10.0  # 結果顯示時間
                    ShowStartTime = time.time()

                    while True:  # 顯示比對結果
                        Ret, NewFrame = cap.read()
                        if not Ret:
                            break

                        newImg, newResults = detector.findPose(NewFrame)
                        newLmlist, newBboxinfo = detector.findPosition(newImg)
                        if newBboxinfo:  # 若有新骨架資訊則繼續傳送
                            newLmString = ''
                            for lm in newLmlist:
                                newLmString += f'{lm[0]},{newImg.shape[0]-lm[1]},{lm[2]},'
                            self.sender.send_motion_data(newLmString)

                        self.draw_info(newImg, self.mode, None)
                        out.write(newImg)
                        cv2.imshow('畫面顯示', newImg)

                        if time.time() - ShowStartTime >= ShowResultDuration:
                            break
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

                    TimeOut = False
                    StartTime = None
                    self.PointsRecord = []
                    ReferenceSequences = None
                    Detect = True

            else:  # 若模式為停止
                StartTime = None
                RestTime = None
                self.message = None
                TimeOut = False
                self.PointsRecord = []
                Detect = True

            if self.message is not None:  # 若有訊息則傳送
                self.sender.send_error_message(self.message)

            out.write(img)  # 寫入影片檔案
            cv2.imshow('畫面顯示', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):  # 按下 'q' 鍵則結束
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.switchSocket.close()  # 關閉 Socket 連接

if __name__ == '__main__':
    args = get_parser()  # 解析命令行參數
    BaseDetect(args).process()  # 啟動主程序

