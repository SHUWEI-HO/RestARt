import cv2
import mediapipe as mp
import argparse
import numpy as np
from Camera import check_camera
from PIL import ImageFont, ImageDraw, Image
from Tool import *
from Motion.HalfSquat import HalfSquat
from Motion.KneeRaise import KneeRaise
from Motion.LungeStretch import LungeStretch
from Motion.ShoulderBladeStretch import ShoulderBladeStretch
from Motion.LateralRaise import LateralRaise
from Motion.Pendulum import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-num', default=4, type=int, help="選擇攝影機的編號")
    parser.add_argument('--exercise-mode', default='HalfSquat', help='運動類型')
    parser.add_argument('--direction', default=None, help='哪一側')
    return parser.parse_args()

class MotionCapture:
    def __init__(self, model_type='Pose'):

        self.args = get_parser()
        self.model_type = model_type
        self.model, self.mp_pose = initial_model(self.model_type)
        self.angle_list = []
        self.counter = 0
        self.stage = None
        self.message = None
        self.status = False
        self.fontpath = "font\msjh.ttc"  
        self.font = ImageFont.truetype(self.fontpath, 24)
        self.CurrentWindow = []
        self.AngleRecord = []

        self.PointsList = []
        self.startn = 1

    def process_mode(self, mode, landmarks, width, height, image, direction=None):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        if mode == 'HalfSquat':

            self.message, self.counter, self.stage, self.status = HalfSquat(landmarks,
                                                                            width,
                                                                            height,
                                                                            self.font,
                                                                            draw,
                                                                            self.counter,
                                                                            self.stage,
                                                                            self.status).process()

        elif mode == 'KneeRaise':
            
            self.message, self.counter, self.stage, self.status = KneeRaise(landmarks,
                                                                            direction,
                                                                            width,
                                                                            height,
                                                                            self.font,
                                                                            draw,
                                                                            self.counter,
                                                                            self.stage,
                                                                            self.status).process()

        elif mode == 'LungeStretch':
            self.message, self.counter, self.stage, self.status = LungeStretch(landmarks,
                                                                               direction,
                                                                               width,
                                                                               height,
                                                                               self.font,
                                                                               draw,
                                                                               self.counter,
                                                                               self.stage,
                                                                               self.status).process()

        elif mode == 'ShoulderBladeStretch':
            self.message, self.counter, self.stage, self.status = ShoulderBladeStretch(landmarks,
                                                                                       width,
                                                                                       height,
                                                                                       self.font,
                                                                                       draw,
                                                                                       self.counter,
                                                                                       self.stage,
                                                                                       self.status).process()

        elif mode == 'LateralRaise':
            self.message, self.counter, self.stage, self.status = LateralRaise(landmarks,
                                                                               direction,
                                                                               width,
                                                                               height,
                                                                               self.font,
                                                                               draw,
                                                                               self.counter,
                                                                               self.stage,
                                                                               self.status).process()

        elif mode == 'Pendulum':
            self.message, self.counter, self.stage, self.status = Pendulum(landmarks,
                                                                           direction,
                                                                           width,
                                                                           height,
                                                                           self.font,
                                                                           draw,
                                                                           self.counter,
                                                                           self.stage,
                                                                           self.status).process()



        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def draw_info(self, image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        draw.text((30, 60), f"完成次數: {self.counter}", font=self.font, fill=(0, 255, 0))
        if self.message is not None:
            color = (0, 255, 0) if self.status else (255, 0, 0)
            draw.text((30, 100), f"訊息: {self.message}", font=self.font, fill=color)

        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def process(self, mode, direction):
        cap = check_camera(self.args.camera_num)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output_video_.mp4', fourcc, 24, size)


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = self.model.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if self.model_type == 'Pose' and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                landmark_coords = get_landmarks(self.mp_pose, landmarks)

                self.process_mode(mode, landmark_coords, width, height, image, direction=direction)
                self.draw_info(image)

                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
            else:
                print("未檢測到任何內容")

            out.write(image)
            cv2.imshow('畫面顯示', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_parser()
    motion_capture = MotionCapture(model_type='Pose')
    motion_capture.process(motion_capture.args.exercise_mode, args.direction)
