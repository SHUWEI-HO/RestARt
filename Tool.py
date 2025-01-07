import cv2
import mediapipe as mp
import numpy as np
import math
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.spatial.distance import euclidean
from dtaidistance.subsequence.dtw import subsequence_alignment


# 取得指定的關節點座標（x, y）
def get_landmarks(mp_pose, landmarks):
    """
    取得所需的關節點座標。
    Args:
        mp_pose: Mediapipe Pose 模組
        landmarks: Mediapipe 偵測出的所有關節點座標

    Returns:
        landmark_coords: 字典形式的關節點座標，格式為 {關節名稱: [x, y]}
    """
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


# 初始化 Mediapipe 模型
def initial_model(model_type):
    """
    初始化 Mediapipe 模型。
    Args:
        model_type: 模型類型，可選 'Pose' 或 'Hands'

    Returns:
        model: Mediapipe 模型
        mp_module: 對應的 Mediapipe 模組
    """
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


# 計算三個點形成的角度
def calculate_angle(point1, point2, point3):
    """
    計算由三個點形成的角度。
    Args:
        point1, point2, point3: 三個點的 [x, y] 座標

    Returns:
        angle: 由三個點形成的角度（整數）
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0]) 
    angle = np.abs(radians * 100.0 / np.pi) 
    angle = 360 - angle if angle > 180 else angle
    angle = float(angle)
    
    return int(round(angle, 2))


# 計算關節點間的距離（正規化）
def compute_distance(landmarks):
    """
    計算關節點之間的距離，並以參考距離進行正規化。
    Args:
        landmarks: Mediapipe 偵測出的所有關節點座標

    Returns:
        Distances: 正規化後的距離列表
    """
    Distances = []
    # 定義要計算的關節對
    Pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (23, 24), 
             (23, 25), (25, 27), (24, 26), (26, 28), (28, 30), (27, 29)]

    # 定義參考點對（用於正規化）
    ReferencePair = (0, 29)

    p_ref1 = np.array([landmarks[ReferencePair[0]].x, landmarks[ReferencePair[0]].y])
    p_ref2 = np.array([landmarks[ReferencePair[1]].x, landmarks[ReferencePair[1]].y])
    reference_distance = np.linalg.norm(p_ref1 - p_ref2)

    # 計算每對關節點的距離，並進行正規化
    for pair in Pairs:
        p1 = np.array([landmarks[pair[0]].x, landmarks[pair[0]].y])
        p2 = np.array([landmarks[pair[1]].x, landmarks[pair[1]].y])
        distance = np.linalg.norm(p1 - p2) / reference_distance
        Distances.append(distance)

    return Distances


# 動態時間規劃（DTW）處理
def DTW_Processing(InputSequence, ReferenceSequence):
    """
    使用動態時間規劃（DTW）處理輸入序列與參考序列。
    Args:
        InputSequence: 輸入的關節點序列
        ReferenceSequence: 參考的關節點序列

    Returns:
        values: 最佳匹配的 DTW 距離值
        paths: 最佳匹配的路徑
        SEpoint: 最佳匹配的起點與終點
    """
    Input = []
    Refer = [] 

    # 計算每個序列的角度
    for InSeq in InputSequence:
        Input.append(calculate_angle(InSeq[0], InSeq[1], InSeq[2]))
    for ReSeq in ReferenceSequence:
        Refer.append(calculate_angle(ReSeq[0], ReSeq[1], ReSeq[2]))

    Input = np.array(Input)
    Refer = np.array(Refer)

    # 使用 subsequence_alignment 找到最佳匹配
    SA = subsequence_alignment(Refer, Input)
    values = []
    paths = []
    SEpoint = []  # 起點與終點
    for kmatch in SA.kbest_matches(3):
        values.append(kmatch.value)
        paths.append(kmatch.path)
        SEpoint.append(kmatch.segment)

    return values, paths, SEpoint


# 計算兩點之間的歐氏距離並檢查是否低於門檻值
def find_point(v1, v2, weight=None):
    """
    檢查兩個點之間的歐氏距離是否小於門檻值。
    Args:
        v1: 偵測動作數據
        v2: 範例起始點數據
        weight: 權重（預設為 1）

    Returns:
        bool: 若距離小於門檻值則回傳 True，否則回傳 False
    """
    assert len(v1) == len(v2), 'Error Contrasting!!!'
    
    threshold = 0.15  # 距離門檻值
    
    if weight is None:
        weight = 1
    TotalEdistance = 0.0
    
    for idx in range(len(v1)):
        TotalEdistance += weight * euclidean(v1[idx], v2[idx])
    
    if TotalEdistance < threshold:
        return True
    
    return False
