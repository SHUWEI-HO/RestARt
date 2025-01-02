import cv2
import mediapipe as mp
import argparse
import numpy as np
from Camera import *
from Tool import *
from PIL import ImageFont, ImageDraw, Image
from sklearn.preprocessing import LabelEncoder
import joblib
from Motion import HalfSquat, KneeRaise, LungeStretch, LateralRaise, ShoulderBladeStretch, Pendulum
import warnings
from Contrast import contrasting
import time
import os



warnings.filterwarnings("ignore", category=UserWarning)

def get_parser():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--camera', default=0, type=int)
    Parser.add_argument('--mode', type=str, default='HalfSquat')
    return Parser.parse_args()

class BaseDetect:
    def __init__(self, Args):
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

        self.Scaler = joblib.load(ScalerChoose[Args.mode])
        self.LabelEncoder = joblib.load(LabelEncChoose[Args.mode])
        # 載入序列資訊
        self.MaxLength, self.FeatureDim = joblib.load(SeqInfoChoose[Args.mode])

        self.Camera = Args.camera
        self.Mode = Args.mode

        # PoseModel 用於 MediaPipe Pose 辨識
        self.PoseModel, self.MpPose = initial_model('Pose')

        # 字型設定
        self.FontPath = 'Font/msjhbd.ttc'
        self.Font = ImageFont.truetype(self.FontPath, 32)

        self.Counter = 0
        self.Message = None
        self.Status = None
        self.EndTime = 7

        # ActionModel 用於分類預測
        self.ActionModel = joblib.load(ModelChoose[self.Mode])

        self.label = '未檢測'
        self.confidence = 0.0

        # 用於累積序列的緩衝區
        self.SequenceBuffer = []

    def draw_info(self, ImageFrame, Mode, RestTime=None, Detect=True):
        PilImage = Image.fromarray(cv2.cvtColor(ImageFrame, cv2.COLOR_BGR2RGB))
        Draw = ImageDraw.Draw(PilImage)

        ModeChoose = {
            'HalfSquat': '半蹲',
            'KneeRaise': '提膝',
            'LungeStretch': '弓箭步',
            'LateralRaise': '手臂側舉',
            'ShoulderBladeStretch': '肩胛拉伸',
            'Pendulum': '鐘擺運動'
        }

        ModeText = f"模式: {ModeChoose[Mode]}"
        TextBbox = Draw.textbbox((0, 0), ModeText, font=self.Font)
        TextWidth = TextBbox[2] - TextBbox[0]
        TextHeight = TextBbox[3] - TextBbox[1]

        OuterMargin = 10
        InnerMargin = 5

        OuterLeft = 20
        OuterTop = 390
        OuterRight = OuterLeft + TextWidth + OuterMargin * 2
        OuterBottom = OuterTop + TextHeight + OuterMargin * 2

        Draw.rectangle([OuterLeft, OuterTop, OuterRight, OuterBottom], fill=(255, 0, 0))

        InnerLeft = OuterLeft + InnerMargin
        InnerTop = OuterTop + InnerMargin
        InnerRight = OuterRight - InnerMargin
        InnerBottom = OuterBottom - InnerMargin

        Draw.rectangle([InnerLeft, InnerTop, InnerRight, InnerBottom], fill=(0, 255, 0))
        Draw.text((InnerLeft + 5, InnerTop + 0), ModeText, font=self.Font, fill=(128, 124, 0))

        # 顯示預測結果
        # PredictionText = f"預測: {self.label} ({round(self.confidence*100,2)}%)"
        # Draw.text((30, 70), PredictionText, font=self.Font, fill=(0, 0, 255))

        # 顯示訊息
        if self.Message is not None:
            Color = (120, 255, 0) if self.Status else (255, 25, 0)
            Draw.text((30, 70), f"訊息 : {self.Message}", font=self.Font, fill=Color)

        # 繪製垂直進度條
        ImageWidth, ImageHeight = PilImage.size
        TotalHeight = 300
        BarWidth = 20
        LeftX = ImageWidth - 30 - BarWidth
        TopY = (ImageHeight - TotalHeight) // 2
        Draw.rectangle([LeftX, TopY, LeftX + BarWidth, TopY + TotalHeight], fill=(200, 200, 200))

        if RestTime is not None:
            Progress = RestTime / self.EndTime
            CurrentHeight = int(TotalHeight * Progress)
            Draw.rectangle([LeftX, TopY + (TotalHeight - CurrentHeight), LeftX + BarWidth, TopY + TotalHeight], fill=(255, 165, 0))
        else:
            Draw.rectangle([LeftX, TopY, LeftX + BarWidth, TopY + TotalHeight], fill=(255, 165, 0))

        ImageFrame[:] = cv2.cvtColor(np.array(PilImage), cv2.COLOR_RGB2BGR)

    def pad_and_flatten(self, Seq):
        CurrentLength = Seq.shape[0]
        if CurrentLength < self.MaxLength:
            PadLength = self.MaxLength - CurrentLength
            Seq = np.vstack([Seq, np.zeros((PadLength, self.FeatureDim))])
        elif CurrentLength > self.MaxLength:
            Seq = Seq[-self.MaxLength:, :]

        Flattened = Seq.reshape(-1)
        Flattened = Flattened.reshape(1, -1)
        return Flattened

    def process(self):
        Cap = check_camera(self.Camera)
        Width = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        Height = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        Size = (Width, Height)
        Fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        Out = cv2.VideoWriter('output_video_.mp4', Fourcc, 24, Size)

        Detect = True
        StartTime = None
        ReferenceSequences = None
        TimeOut = False
        RestTime = None
        Started = False
        Countdown = 3
        CountdownStartTime = None

        while True:
            self.time = None
            Ret, Frame = Cap.read()
            if not Ret:
                break

            # 尚未開始, 顯示開始提示
            if not Started:
                ImageFrame = Frame.copy()
                PilImage = Image.fromarray(cv2.cvtColor(ImageFrame, cv2.COLOR_BGR2RGB))
                Draw = ImageDraw.Draw(PilImage)
                Draw.text((30, 30), "按空白鍵開始", font=self.Font, fill=(0, 255, 0))

                # 倒數計時
                if CountdownStartTime is not None:
                    Elapsed = time.time() - CountdownStartTime
                    Remaining = Countdown - int(Elapsed)
                    if Remaining > 0:
                        Draw.text((30, 80), f"倒數 {Remaining} 秒後開始", font=self.Font, fill=(70, 255, 100))
                    else:
                        Started = True
                        CountdownStartTime = None

                ImageFrame[:] = cv2.cvtColor(np.array(PilImage), cv2.COLOR_RGB2BGR)
                cv2.imshow('畫面顯示', ImageFrame)

                Key = cv2.waitKey(1) & 0xFF
                if Key == ord(' '):
                    CountdownStartTime = time.time()
                elif Key == ord('q'):
                    break
                continue

            else:
                # 已開始，進入偵測
                if Detect:
                    Ret, Frame = Cap.read()
                    if not Ret:
                        break

                    ImageFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
                    ImageFrame.flags.writeable = False
                    Results = self.PoseModel.process(ImageFrame)
                    ImageFrame.flags.writeable = True
                    ImageFrame = cv2.cvtColor(ImageFrame, cv2.COLOR_RGB2BGR)

                    if Results.pose_landmarks:
                        Landmarks = Results.pose_landmarks
                        Distances = compute_distance(Landmarks.landmark)
                        Distances = np.array(Distances).reshape(1, -1)
                        self.SequenceBuffer.append(Distances[0])

                        SeqArray = np.array(self.SequenceBuffer)
                        FlattenedSeq = self.pad_and_flatten(SeqArray)

                        XInput = self.Scaler.transform(FlattenedSeq)
                        Probabilities = self.ActionModel.predict_proba(XInput)
                        Prediction = np.argmax(Probabilities, axis=1)[0]
                        Confidence = Probabilities[0, Prediction]
                        Label = self.LabelEncoder.inverse_transform([Prediction])[0]

                        self.label, self.confidence = Label, Confidence

                        # if Label == self.Mode:
                        if StartTime is None:
                            StartTime = time.time()
                            self.PointsRecord = []
                        ElapsedTime = time.time() - StartTime

                        if ElapsedTime >= self.EndTime:
                            RestTime = None
                            TimeOut = True
                            self.Message = '動作完成!即將進行比對'
                            self.status = True

                            # =============================
                            # 新增：先顯示「動作完成!即將進行比對」的暫停畫面 N 秒
                            ShowMessageDuration = 3.0  # 顯示秒數可自訂
                            ShowMessageStart = time.time()

                            while time.time() - ShowMessageStart < ShowMessageDuration:
                                ret_msg, frame_msg = Cap.read()
                                if not ret_msg:
                                    break
                                tempFrame = frame_msg.copy()
                                self.draw_info(tempFrame, self.Mode, None, Detect=False)
                                Out.write(tempFrame)
                                cv2.imshow('畫面顯示', tempFrame)
                                if cv2.waitKey(10) & 0xFF == ord('q'):
                                    break
                            # =============================

                        else:
                            RestTime = int(self.EndTime - ElapsedTime)
                            self.Message = '請完成動作~'

                        LandmarksCoords = get_landmarks(self.MpPose, Landmarks.landmark)
                        if self.Mode == 'HalfSquat':
                            self.PointsRecord, ReferenceSequences = HalfSquat.HalfSquat(LandmarksCoords).process(TimeOut, self.PointsRecord)
                        elif self.Mode == 'KneeRaise':
                            self.PointsRecord, ReferenceSequences = KneeRaise.KneeRaise(LandmarksCoords).process(TimeOut, self.PointsRecord)
                        elif self.Mode == 'LateralRaise':
                            self.PointsRecord, ReferenceSequences = LateralRaise.LateralRaise(LandmarksCoords).process(TimeOut, self.PointsRecord)
                        elif self.Mode == 'ShoulderBladeStretch':
                            self.PointsRecord, ReferenceSequences = ShoulderBladeStretch.ShoulderBladeStretch(LandmarksCoords).process(TimeOut, self.PointsRecord)
                        # else:
                        #     StartTime = None
                        #     TimeOut = False
                        #     self.Message = '請做出正確動作!!!'
                    else:
                        StartTime = None
                        TimeOut = False
                        self.Message = '未檢測到人體姿勢'

                    self.draw_info(ImageFrame, self.Mode, RestTime)
                    # mp.solutions.drawing_utils.draw_landmarks(
                    #     ImageFrame, Results.pose_landmarks, self.MpPose.POSE_CONNECTIONS,
                    #     mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    #     mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    # )

                    # TimeOut 且成功取到 ReferenceSequences 才進到下一階段（對比結果）
                    if TimeOut and ReferenceSequences is not None:
                        Detect = False

                else:
                    # ---- (Detect = False) ----
                    # 進行 contrasting
                    self.Message, self.Status = contrasting(self.PointsRecord, ReferenceSequences, self.Mode)
                    DisplayedMessage = self.Message
                    DisplayedStatus = self.Status

                    ShowResultDuration = 10.0  # 比對後的顯示秒數
                    ShowStartTime = time.time()

                    while True:
                        Ret, NewFrame = Cap.read()
                        if not Ret:
                            break

                        ImageFrame = NewFrame.copy()
                        # if Results.pose_landmarks:
                            # mp.solutions.drawing_utils.draw_landmarks(
                            #     ImageFrame,
                            #     Results.pose_landmarks,
                            #     self.MpPose.POSE_CONNECTIONS,
                            #     mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            #     mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                            # )

                        self.Message = DisplayedMessage
                        self.Status = DisplayedStatus

                        self.draw_info(ImageFrame, self.Mode, None, False)

                        Out.write(ImageFrame)
                        cv2.imshow('畫面顯示', ImageFrame)

                        if time.time() - ShowStartTime >= ShowResultDuration:
                            break
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

                    # 結束顯示後，重置並準備下一輪偵測
                    self.PointsRecord, ReferenceSequences = [], None
                    TimeOut, StartTime = False, None
                    Detect = True

            Out.write(ImageFrame)
            cv2.imshow('畫面顯示', ImageFrame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        Cap.release()
        Out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Args = get_parser()
    BaseDetect(Args).process()
