import cv2
import mediapipe as mp
import numpy as np
import time
import os
import argparse
from Tool import *
from Camera import *
from PIL import Image, ImageDraw, ImageFont


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name')
    parser.add_argument('--save-dir', default='Dataset')
    parser.add_argument('--model-type', default='Pose')
    parser.add_argument('--motion', default='HalfSquat')
    parser.add_argument('--camera', type=int, default=1)

    return parser.parse_args()


class DataCollect:
    def __init__(self, FileName, OutDir, ModelType):
        self.FileName = FileName
        self.OutDir = OutDir
        self.OutPath = os.path.join(self.OutDir, 'Points',self.FileName)
        self.VideosPath = os.path.join(self.OutDir, 'Videos')
        self.DistancePath = os.path.join(self.OutDir, 'Distances', self.FileName)
        self.model, _ = initial_model(ModelType)
        self.mp = mp  
    def pointCollect(self, landmarks, Mode):
        ModeType = ['HalfSquat', 'KneeRaise', 'LungeStretch', 'ShoulderBladeStretch', 'LateralRaise', 'Pendulum']
        Point = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]
        assert Mode in ModeType, 'Please enter correct motion!'
        PointDataStoration = []
        for point in Point:
            X = landmarks[point].x
            Y = landmarks[point].y
            PointDataStoration.append(np.array([X, Y]))
        return PointDataStoration

    def compute_distance(self, landmarks):

        Distances = []
        
        Pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (23, 24), 
                 (23, 25), (25, 27), (24, 26), (26, 28), (28, 30), (27, 29)]

        ReferencePair = (0, 29)

        p_ref1 = np.array([landmarks[ReferencePair[0]].x, landmarks[ReferencePair[0]].y])
        p_ref2 = np.array([landmarks[ReferencePair[1]].x, landmarks[ReferencePair[1]].y])
        reference_distance = np.linalg.norm(p_ref1 - p_ref2)

        for pair in Pairs:
            p1 = np.array([landmarks[pair[0]].x, landmarks[pair[0]].y])
            p2 = np.array([landmarks[pair[1]].x, landmarks[pair[1]].y])
            distance = np.linalg.norm(p1 - p2) / reference_distance
            Distances.append(distance)

        return Distances
            
    def draw_chinese_text(self, img, text, position, textColor=(255, 255, 255), textSize=30):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('font/msjh.ttc', textSize, encoding='utf-8')
        draw.text(position, text, fill=textColor, font=font)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def process(self, index, mode):
        os.makedirs(os.path.join(self.OutDir), exist_ok=True)
        os.makedirs(os.path.join(self.OutDir, 'Points'), exist_ok=True)
        os.makedirs(os.path.join(self.OutDir, 'Distances'), exist_ok=True)
        os.makedirs(os.path.join(self.OutDir, 'Videos'), exist_ok=True)

        cap = check_camera(index)
        DataCollection, Distances = [], []
        Collect = False
        Preparing = False
        PrepareStartTime = None
        StartTime = None

        Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        Size = (Width, Height)
        Fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        Out = cv2.VideoWriter(os.path.join(self.VideosPath, self.FileName+'_video.mp4'), Fourcc, 24, Size)


        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if not Collect and not Preparing:
                frame = self.draw_chinese_text(frame, "按空格鍵開始收集", (50, 50), textColor=(0, 255, 0), textSize=32)
            elif Preparing:
                ElapsedPrepareTime = int(time.time() - PrepareStartTime)
                RemainingPrepareTime = 3 - ElapsedPrepareTime 
                if RemainingPrepareTime >= 0:
                    frame = self.draw_chinese_text(frame, f"準備時間：{RemainingPrepareTime} 秒", (50, 50), textColor=(0, 255, 255), textSize=32)
                else:
                    Collect = True
                    StartTime = time.time()
                    Preparing = False
            elif Collect:
                ElapsedTime = int(time.time() - StartTime)
                RemainingTime = 8 - ElapsedTime
                frame = self.draw_chinese_text(frame, f"剩餘時間：{RemainingTime} 秒", (50, 50), textColor=(0, 0, 255), textSize=32)
                if ElapsedTime >= 8:
                    break

            RGBFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Results = self.model.process(RGBFrame)

            if Results.pose_landmarks:
                self.mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    Results.pose_landmarks,
                    self.mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )

            if Results.pose_landmarks and Collect:
                landmarks = Results.pose_landmarks.landmark
                Points = self.pointCollect(landmarks, mode)
                Distance = self.compute_distance(landmarks)
                DataCollection.append(Points)
                Distances.append(Distance)

            cv2.imshow("資料收集", frame)
            key = cv2.waitKey(1)

            Out.write(frame)
            if key == 32 and not Collect and not Preparing:
                Preparing = True
                PrepareStartTime = time.time()
            elif key == ord('q') or key == ord('Q'):
                break
                
        cap.release()
        Out.release()
        cv2.destroyAllWindows()
        np.save(self.OutPath, np.array(DataCollection))
        np.save(self.DistancePath, np.array(Distances))
        print(f'Points Data saved to {self.OutPath}\n')
        print(f'Distances Data saved to {self.DistancePath}')


def main():
    args = get_parser()
    FilePath = args.file_name
    OutputPath = args.save_dir
    ModelName = args.model_type
    ExerciseMotion = args.motion
    CameraIndex = args.camera


    DC = DataCollect(FilePath, OutputPath, ModelName)
    DC.process(CameraIndex, ExerciseMotion)


if __name__ == '__main__':
    main()
