import cv2
import mediapipe as mp
import numpy as np
import math
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.spatial.distance import euclidean
from dtaidistance.subsequence.dtw import subsequence_alignment


def get_landmarks(mp_pose, landmarks):

    required_landmarks = {
        'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
        'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST,
        'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST,
        'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
        'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
        'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
        'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
        'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
        'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE
    }
    
    landmark_coords = {}
    for name, idx in required_landmarks.items():
        landmark = landmarks[idx.value]
        landmark_coords[name] = [landmark.x, landmark.y]

    return landmark_coords


def initial_model(model_type):

    assert model_type in ['Pose', 'Hands']

    if model_type == 'Pose':
        mp_pose = mp.solutions.pose
        model = mp_pose.Pose(static_image_mode=False,
                                        model_complexity=2,
                                        smooth_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
        return model, mp_pose
    
    elif model_type == 'Hands':
        mp_hands = mp.solutions.hands
        model = mp_hands.Hands(static_image_mode=False,
                                            model_complexity=2,
                                            smooth_landmarks=True,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)
        return model, mp_hands



def calculate_angle(point1, point2, point3):

    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    radians = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0]) 
    angle = np.abs(radians * 100.0 / np.pi) 
    angle = 360 - angle if angle > 180 else angle
    angle = float(angle)
    
    return int(round(angle, 2))

def compute_distance(landmarks):

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
            distance = np.linalg.norm(p1 - p2)/reference_distance 
            Distances.append(distance)

        return Distances

def DTW_Processing(InputSequence, ReferenceSequence):

    Input = []
    Refer = [] 

    for InSeq in InputSequence:

        Input.append(calculate_angle(InSeq[0], InSeq[1], InSeq[2]))
    for ReSeq in ReferenceSequence:

        Refer.append(calculate_angle(ReSeq[0], ReSeq[1], ReSeq[2]))


    Input = np.array(Input)
    Refer = np.array(Refer)
    

    SA = subsequence_alignment(Refer, Input)
    values = []
    paths = []
    SEpoint = [] # start end point
    for kmatch in SA.kbest_matches(3):
        values.append(kmatch.value)
        paths.append(kmatch.path)
        SEpoint.append(kmatch.segment)



    # distance, paths = dtw.warping_paths(Input, Refer)
    # paths = dtw.best_path(paths)
    # breakpoint()
    return values, paths, SEpoint


def find_point(v1, v2, weight=None):
    
    '''
    v1:偵測動作數據
    v2:範例起始點數據
    '''
    assert len(v1)==len(v2), 'Error Contrasting!!!'
    
    threshold = 0.15
    
    
    if weight is None:
        weight = 1
    TotalEdistance = 0.0
    
    for idx in range(len(v1)):
        TotalEdistance += weight * euclidean(v1[idx], v2[idx])
    
    if TotalEdistance < threshold:
        
        return True
    
    return False



def calculate_angle(point1, point2, point3):

    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    radians = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0]) 
    angle = np.abs(radians * 100.0 / np.pi) 
    angle = 360 - angle if angle > 180 else angle
    angle = float(angle)
    
    return int(round(angle, 2))