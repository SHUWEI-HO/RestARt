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

warnings.filterwarnings("ignore", category=UserWarning)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default=0, type=int)
    parser.add_argument('--mode', type=str, default='HalfSquat')
    return parser.parse_args()

# ------------------------------------------------------------
# 專門負責傳遞【骨架動作(5054)】與【錯誤訊息(5055)】的類別
# ------------------------------------------------------------
class MotionErrorSender:
    def __init__(self):
        # (1) 動作數據傳輸：5054
        self.motionSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.motionAddressPort = ("127.0.0.1", 6720)

        # (2) 錯誤訊息傳輸：5055
        self.errorSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.errorAddressPort = ("127.0.0.1", 7801)

        self.timeSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.timeAddressPort = ("127.0.0.1", 8106)

    def send_motion_data(self, data):
        # data 為骨架動作資料字串
        self.motionSocket.sendto(data.encode(), self.motionAddressPort)

    def send_error_message(self, message):
        # message 為錯誤/提示訊息
        self.errorSocket.sendto(message.encode(), self.errorAddressPort)
    
    def send_time(self, time):
        self.timeSocket.sendto(time.encode(), self.timeAddressPort)
# ------------------------------------------------------------
# 主程式：包含「攝影機擷取、骨架偵測、動作判斷、錯誤訊息、動作切換(5053)」
# ------------------------------------------------------------
class BaseDetect:
    def __init__(self, args):
        # 1) 先保存初始參數 (Camera 與 預設動作 mode)
        self.Camera = args.camera
        self.mode = args.mode

        # 2) 建立「接收動作編號」的 socket (5053)
        #    用來接收 Unity 按鈕觸發後送來的「動作切換」指令
        self.switchSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.switchSocket.bind(("0.0.0.0", 5053))  # 監聽 5053
        self.switchSocket.setblocking(False)       # 設定非阻塞



        # 3) 載入初始模型（依 self.mode)
        self.load_model_scaler_label(self.mode)

        # 4) Mediapipe Pose 初始化
        self.model, self.mp_pose = initial_model('Pose')

        # 5) 繪製字型設定 (請改成您實際擁有的字體路徑)
        self.fontpath = 'Font/msjhbd.ttc'
        self.font = ImageFont.truetype(self.fontpath, 32)

        # 6) 用於動作預測 / 骨架傳輸
        self.sequence_buffer = []  # 動作模型預測序列
        self.posList = []          # 用於平滑後傳到 Unity 的骨架

        # 7) 狀態參數
        self.counter = 0
        self.message = None
        self.status = None
        self.EndTime = 7
        self.label = None
        self.confidence = 0

        # 8) 動作比對 / 錯誤訊息
        self.PointsRecord = []
        self.sender = MotionErrorSender()
        
        self.label = '未檢測'
        self.confidence = 0.0

    # --------------------------------------------------------
    # 載入 / 切換模型
    # --------------------------------------------------------
    def load_model_scaler_label(self, mode_name):
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

        # 確認輸入的 mode_name 合法
        if mode_name not in ModelChoose:
            raise ValueError(f"未知的模式: {mode_name}")

        self.mode = mode_name
        self.Model = joblib.load(ModelChoose[mode_name])
        self.Scaler = joblib.load(ScalerChoose[mode_name])
        self.LabelEncoder = joblib.load(LabelEncChoose[mode_name])
        self.max_length, self.feature_dim = joblib.load(SeqInfoChoose[mode_name])

        print(f"[Python] 已切換 / 載入模式: {mode_name}")

    # --------------------------------------------------------
    # 在每幀迴圈開頭嘗試接收「Unity 傳來的動作編號」
    # --------------------------------------------------------
    def try_receive_mode_switch(self, start):
        # print(start)
        ACTION_MAP = {
            '2': 'HalfSquat',
            '3': 'KneeRaise',
            '5': 'ShoulderBladeStretch',
            '6': 'LateralRaise',
            '8': 'Close'
        }
        try:
            data, addr = self.switchSocket.recvfrom(1024)
            message = data.decode('utf-8').strip()

            message = message.split(',')
            # breakpoint()
            action_number = message[0].strip()
            start_mode = message[1].strip()
            print(f"[Python] 收到動作編號: {action_number}")
            print(f"[Python] 收到啟動信息: {start_mode}")

            if action_number in ACTION_MAP:
                new_mode = ACTION_MAP[action_number]
                if action_number != '8':
                    self.load_model_scaler_label(new_mode)
                print(start_mode)
                return True, start_mode
            else:
                print(f"[Python] 未知的動作編號: {action_number}")
                #TODO 這裡還要寫一個continue 
        except BlockingIOError:
            pass  # 這一幀沒有收到資料就直接跳過
        

        return None, start

    # --------------------------------------------------------
    # 在畫面上繪製相關資訊 (模式、預測、訊息、進度條)
    # --------------------------------------------------------
    def draw_info(self, image, mode, RestTime=None):
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
        cap = check_camera(self.Camera)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output_video_.mp4', fourcc, 24, size)
        Detect = True
        StartTime = None
        TimeOut = False
        RestTime = None
        detector = PoseDetector()

        ReferenceSequences = None  # 用來記錄動作完成時要比對的基準序列
        start_mode = 'Stop'

        while True:
            # (A) 先嘗試看有無新的動作切換
            check, start_mode = self.try_receive_mode_switch(start_mode)
            # print(start_mode)
            # breakpoint()


            if check:
                Detect = check

            ##### 修改處 (1)：
            # 把「讀取畫面 + 骨架偵測 + 傳骨架資料」的動作抽到外面
            # 讓它不管 Detect 是 True 或 False 都會做。
            ret, frame = cap.read()
            if not ret:
                break

            # 先用 cvZone PoseDetector 擷取骨架
            img, results = detector.findPose(frame)
            lmlist, bboxinfo = detector.findPosition(img)

            # 若偵測到有人體
            if bboxinfo:
                landmarks = results.pose_landmarks.landmark
                # 建立骨架字串
                lmString = ''
                for lm in lmlist:
                    lmString += f'{lm[0]},{img.shape[0]-lm[1]},{lm[2]},'
                self.posList.append(lmString)

                # 每 5 幀做一次平滑並傳送
                if len(self.posList) >= 5:
                    avgPos = []
                    num_lm = 33  # Mediapipe Pose 預設 33 個關節
                    for i in range(num_lm):
                        xPos, yPos, zPos = [], [], []
                        for j in range(len(self.posList) - 2, len(self.posList) + 3):
                            if 0 <= j < len(self.posList):
                                coords = self.posList[j].split(',')
                                if i*3 + 2 < len(coords):
                                    xPos.append(float(coords[i*3]))
                                    yPos.append(float(coords[i*3 + 1]))
                                    zPos.append(float(coords[i*3 + 2]))
                        avgX = sum(xPos) / len(xPos)
                        avgY = sum(yPos) / len(yPos)
                        avgZ = sum(zPos) / len(zPos)
                        avgPos.append(f'{avgX},{avgY},{avgZ},')
                    avgPosString = ''.join(avgPos)
                    self.sender.send_motion_data(avgPosString)  # 持續傳骨架資料

            # 如果目前可以偵測動作
            if start_mode == 'Going':

                if Detect:
                    # --- 以下是原本 if Detect 區塊裡的流程 ---
                    # (C) 若有人體骨架 -> 進行動作判斷、計時、顯示
                    if bboxinfo:
                        # 以下邏輯跟原本程式相同，做動作判定、計時與 TimeOut ...
                        if StartTime is None:
                            StartTime = time.time()
                            self.PointsRecord = []

                        ElapsedTime = time.time() - StartTime
                        if ElapsedTime >= self.EndTime:
                            RestTime = None
                            TimeOut = True
                            self.status = True
                            self.message = '動作完成! 即將進行比對...'

                            # =============================
                            # 新增：先顯示「動作完成!即將進行比對」的暫停畫面 N 秒
                            ShowMessageDuration = 5.0  # 顯示秒數可自訂
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
                            # =============================
                        else:
                            RestTime = int(self.EndTime - ElapsedTime)
                            self.sender.send_time(str(RestTime))
                            self.message = '請保持動作~'

                        # 分動作處理
                        if bboxinfo:
                            landmarks_coords = get_landmarks(self.mp_pose, landmarks)
                            if self.mode == 'HalfSquat':
                                self.PointsRecord, ReferenceSequences = HalfSquat.HalfSquat(landmarks_coords).process(TimeOut, self.PointsRecord)
                            elif self.mode == 'KneeRaise':
                                self.PointsRecord, ReferenceSequences = KneeRaise.KneeRaise(landmarks_coords).process(TimeOut, self.PointsRecord)
                            elif self.mode == 'LateralRaise':
                                self.PointsRecord, ReferenceSequences = LateralRaise.LateralRaise(landmarks_coords).process(TimeOut, self.PointsRecord)
                            elif self.mode == 'ShoulderBladeStretch':
                                self.PointsRecord, ReferenceSequences = ShoulderBladeStretch.ShoulderBladeStretch(landmarks_coords).process(TimeOut, self.PointsRecord)

                    else:
                        # (c) 未檢測到人體
                        StartTime = None
                        RestTime = None
                        TimeOut = False
                        self.message = '未檢測到人體姿勢'
                        self.status = False

                    self.draw_info(img, self.mode, RestTime)

                    # (D) 若動作超時完成 -> 進入對比階段
                    if TimeOut and ReferenceSequences is not None:
                        Detect = False

                else:
                    ##### 修改處 (2)：
                    # 即便 Detect = False（進入比對顯示階段），也會繼續接收畫面與送出骨架資料。
                    # 下方流程主要是對動作比對結果做暫停顯示，但我們可以把發送骨架的邏輯保留在最上面，
                    # 所以骨架傳輸依舊不會斷。
                    self.message, self.status = contrasting(self.PointsRecord, ReferenceSequences, self.mode)
                    self.sender.send_error_message(self.message)  # 只要對完一次就送一次

                    ShowResultDuration = 10.0
                    ShowStartTime = time.time()

                    while True:
                        Ret, NewFrame = cap.read()
                        if not Ret:
                            break

                        # 這裡一樣做骨架偵測、傳輸（若需要的話）
                        newImg, newResults = detector.findPose(NewFrame)
                        newLmlist, newBboxinfo = detector.findPosition(newImg)
                        if newBboxinfo:
                            newLmString = ''
                            for lm in newLmlist:
                                newLmString += f'{lm[0]},{newImg.shape[0]-lm[1]},{lm[2]},'
                            self.sender.send_motion_data(newLmString)

                        # 顯示比對後的結果
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
            
            else :
                # print(True)
                StartTime = None
                RestTime = None
                self.message = None
                TimeOut = False
                self.PointsRecord = []
                Detect = True

            if self.message is not None:
                self.sender.send_error_message(self.message)


            # 寫在畫面上
            out.write(img)
            cv2.imshow('畫面顯示', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.switchSocket.close()

if __name__ == '__main__':
    args = get_parser()
    BaseDetect(args).process()
