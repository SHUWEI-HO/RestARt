from Tool import DTW_Processing, calculate_angle

def contrasting(InputSequence, ReferenceSequence, mode, status=False):
    """
    利用 DTW_Processing 將使用者輸入序列與參考序列做動作比對。
    若距離 > threshold 則代表動作尚未達成要求，需計算角度差。
    若距離 <= threshold 則代表動作完成。
    
    參數:
        InputSequence     : 使用者實際動作的序列資料 (list of lists)
        ReferenceSequence : 參考的標準動作序列 (list of lists)
        mode              : 動作模式 (string)，可選 'HalfSquat'、'KneeRaise'、'LateralRaise' 或 'ShoulderBladeStretch'
        status            : 動作成功/失敗狀態，預設為 False (bool)
    
    回傳:
        message (string)  : 返回動作比對的提示訊息
        status (bool)     : 動作成功/失敗的狀態值
    """

    # 確認輸入資料是否為空
    assert InputSequence is not None, 'Input is None!!!'
    assert ReferenceSequence is not None, 'Reference is None!!!'

    # 進行 DTW 動作序列比對，取得距離、最佳路徑與匹配點
    Distances, Paths, Points = DTW_Processing(InputSequence, ReferenceSequence)

    threshold = 0.5  # 定義距離的門檻值
    time = len(Distances)  # 計算動作次數

    # 儲存比對結果訊息
    message_store = [f'系統偵測到了{time}次動作:']

    if time != 0:
        for idx, (dis, pth, pnt) in enumerate(zip(Distances, Paths, Points)):
            num = idx + 1  # 動作序號（1-based index）

            # 若距離超過門檻值，表示動作未達標準
            if dis > threshold:
                if mode in ['HalfSquat']:
                    # 深蹲動作比對
                    # 1. 計算每幀的角度，取最小角度(TargetAngle)與其索引(TargetIndex)
                    AngleList = [calculate_angle(points[0], points[1], points[2]) for points in InputSequence[pnt[0]:pnt[1]+1]]
                    TargetAngle = min(AngleList)
                    TargetIndex = AngleList.index(TargetAngle)

                    # 2. 搜尋 Path 中該索引對應的參考動作點位，並儲存至 PathRecord
                    PathRecord = []
                    for p in pth:
                        if p[0] == TargetIndex and p[1] not in PathRecord:
                            if p[1] < len(ReferenceSequence):  # 確保索引不超出範圍
                                try:
                                    PathRecord.append(ReferenceSequence[p[1]])
                                except:
                                    continue

                    # 3. 計算參考點位的最小角度(ExampleAngle)
                    FindAngle = [calculate_angle(pt[0], pt[1], pt[2]) for pt in PathRecord]
                    if len(FindAngle) > 0:
                        ExampleAngle = min(FindAngle)
                    else:
                        ExampleAngle = TargetAngle  # 若 PathRecord 為空則避免報錯，將目標角度作為參考角度

                    # 4. 計算兩者角度差 (Degree)
                    Degree = abs(int(ExampleAngle - TargetAngle))

                    # 根據角度差輸出不同的提示訊息
                    if Degree > 0 and Degree <= 10:
                        message = f'第{num}個動作不正確，請您再稍微蹲低一點'
                    elif Degree > 10 and Degree <= 30:
                        message = f'第{num}個動作不正確，您還需要蹲更低'
                    elif Degree > 30:
                        message = f'第{num}個動作並未達到深蹲動作標準，深蹲動作大腿應和地面保持平行'
                    elif Degree < 0:
                        message = f'第{num}個動作蹲太低囉!請用標準動作進行復健以免受傷'
                    elif Degree == 0:
                        message = f'第{num}次動作符合標準!'
                    message_store.append(message)

                elif mode in ['KneeRaise']:
                    # 提膝動作比對，邏輯與 HalfSquat 類似
                    AngleList = [calculate_angle(points[0], points[1], points[2]) for points in InputSequence[pnt[0]:pnt[1]+1]]
                    TargetAngle = min(AngleList)
                    TargetIndex = AngleList.index(TargetAngle)

                    PathRecord = []
                    for p in pth:
                        if p[0] == TargetIndex and p[1] not in PathRecord:
                            if p[1] < len(ReferenceSequence):
                                try:
                                    PathRecord.append(ReferenceSequence[p[1]])
                                except:
                                    continue

                    FindAngle = [calculate_angle(pt[0], pt[1], pt[2]) for pt in PathRecord]
                    ExampleAngle = min(FindAngle) if len(FindAngle) > 0 else TargetAngle

                    Degree = abs(int(ExampleAngle - TargetAngle))

                    if Degree > 0 and Degree <= 10:
                        message = f'第{num}個動作不正確，請您把腳稍微再提高一點'
                    elif Degree > 10 and Degree <= 30:
                        message = f'第{num}個動作不正確，您還需要把腳提的更高'
                    elif Degree > 30:
                        message = f'第{num}個動作並未達到提膝動作標準，請將膝蓋提高至腰部位置'
                    elif Degree < 0:
                        message = f'第{num}個動作抬太高囉!請用標準動作進行復健以免受傷'
                    elif Degree == 0:
                        message = f'第{num}次動作符合標準!'
                    message_store.append(message)

                elif mode in ['LateralRaise']:
                    # 側舉動作比對，與上述動作類似，取最大角度
                    AngleList = [calculate_angle(points[0], points[1], points[2]) for points in InputSequence[pnt[0]:pnt[1]+1]]
                    TargetAngle = max(AngleList)
                    TargetIndex = AngleList.index(TargetAngle)

                    PathRecord = []
                    for p in pth:
                        if p[0] == TargetIndex:
                            if p[1] < len(ReferenceSequence):
                                try:
                                    PathRecord.append(ReferenceSequence[p[1]])
                                except:
                                    continue

                    ExampleAngle = calculate_angle(PathRecord[0][0], PathRecord[0][1], PathRecord[0][2]) if len(PathRecord) > 0 else TargetAngle

                    Degree = abs(int(ExampleAngle - TargetAngle))

                    if Degree > 0 and Degree <= 10:
                        message = f'第{num}個動作不正確，請您將手再稍微舉高一點'
                    elif Degree > 10 and Degree <= 30:
                        message = f'第{num}個動作不正確，您需要把手舉的更高'
                    elif Degree > 30:
                        message = f'第{num}個動作並未達到手臂側舉動作標準，請將手臂盡量舉高舉直，平行頸部'
                    elif Degree == 0:
                        message = f'第{num}次動作符合標準!'
                    message_store.append(message)
            else:
                # 若距離小於等於門檻值，表示動作成功
                status = True
                message_store.append(f'第{num}次動作正確，符合標準!')

        # 組合所有訊息為一個字串
        message = ''.join([f'/{m}' if i > 0 else m for i, m in enumerate(message_store)])
    else:
        # 若沒有檢測到任何動作，返回失敗訊息
        status = False
        message = '動作不正確，請您做出指定的動作><'
    
    # print(message)  # 輸出訊息
    return message, status
