from Tool import DTW_Processing, calculate_angle

def contrasting(InputSequence, ReferenceSequence, mode, status=False):
    """
    利用 DTW_Processing 將使用者輸入序列與參考序列做動作比對。
    若距離 > threshold 則代表動作尚未達成要求，需計算角度差。
    若距離 <= threshold 則代表動作完成。
    
    參數:
        InputSequence     : 使用者實際動作的序列資料
        ReferenceSequence : 參考的標準動作序列
        mode              : 動作模式 (例如: 'HalfSquat' 或 'KneeRaise' 等)
        status            : 最終返回的動作成功/失敗狀態 (Bool)
    
    回傳:
        message, status   : 返回提示訊息、以及 True/False 狀態值
    """

    assert InputSequence is not None, 'Input is None!!!'
    assert ReferenceSequence is not None, 'Reference is None!!!'

    Distances, Paths, Points = DTW_Processing(InputSequence, ReferenceSequence)
    threshold = 0.5
    time = len(Distances)

    message_store = [f'系統偵測到了{time}次動作:']
    if time != 0:
        for idx, (dis, pth, pnt) in enumerate(zip(Distances, Paths, Points)):
            num = idx+1
            if dis > threshold:
                # if idx == 0 :
                #     message_store.append('動作不正確')
                if mode in ['HalfSquat']:
                    # 1. 計算每幀的角度，取最小角度(TargetAngle)與其索引(index)
                    AngleList = [calculate_angle(points[0], points[1], points[2]) for points in InputSequence[pnt[0]:pnt[1]+1]]
                    TargetAngle = min(AngleList)
                    TargetIndex = AngleList.index(TargetAngle)

                    # 2. 搜尋 Path 中該索引對應的參考動作點位
                    PathRecord = []
                    for p in pth:
                        # p 為 (使用者動作位置, 參考動作位置)
                        if p[0] == TargetIndex and p[1] not in PathRecord:
                            if p[1] < len(ReferenceSequence):  # 若未超出範圍
                                try:
                                    PathRecord.append(ReferenceSequence[p[1]])
                                except:
                                    continue

                    # 3. 計算參考中對應所有角度，取最小角度 (ExampleAngle)
                    FindAngle = [calculate_angle(pt[0], pt[1], pt[2]) for pt in PathRecord]
                    if len(FindAngle) > 0:
                        ExampleAngle = min(FindAngle)
                    else:
                        ExampleAngle = TargetAngle  # 若 PathRecord 為空，避免下方報錯

                    # 4. 計算兩者角度差
                    Degree = abs(int(ExampleAngle - TargetAngle))
                    
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
                    # 1. 計算每幀的角度，取最小角度(TargetAngle)與其索引(index)
                    AngleList = [calculate_angle(points[0], points[1], points[2]) for points in InputSequence[pnt[0]:pnt[1]+1]]
                    TargetAngle = min(AngleList)
                    TargetIndex = AngleList.index(TargetAngle)

                    # 2. 搜尋 Path 中該索引對應的參考動作點位
                    PathRecord = []
                    for p in pth:
                        # p 為 (使用者動作位置, 參考動作位置)
                        if p[0] == TargetIndex and p[1] not in PathRecord:
                            if p[1] < len(ReferenceSequence):  # 若未超出範圍
                                try:
                                    PathRecord.append(ReferenceSequence[p[1]])
                                except:
                                    continue

                    # 3. 計算參考中對應所有角度，取最小角度 (ExampleAngle)
                    FindAngle = [calculate_angle(pt[0], pt[1], pt[2]) for pt in PathRecord]
                    if len(FindAngle) > 0:
                        ExampleAngle = min(FindAngle)
                    else:
                        ExampleAngle = TargetAngle  # 若 PathRecord 為空，避免下方報錯

                    # 4. 計算兩者角度差
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
                    # 若 mode 為其他動作 (如 LateralRaise, ShoulderBladeStretch...)
                    # 依照邏輯改為取最大角度
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

                    if len(PathRecord) > 0:
                        ExampleTargetPoints = max(PathRecord)  # PathRecord 為陣列比較時會比對第一維
                        ExampleAngle = calculate_angle(ExampleTargetPoints[0], ExampleTargetPoints[1], ExampleTargetPoints[2])
                    else:
                        ExampleAngle = TargetAngle

                    Degree = abs(int(ExampleAngle - TargetAngle))
                    if Degree > 0 and Degree <= 10:
                        message = f'第{num}個動作不正確，請您將手再稍微舉高一點'
                    elif Degree > 10 and Degree <= 30:
                        message = f'第{num}個動作不正確，您需要把手舉的更高'
                    elif Degree > 30:
                        message = f'第{num}個動作並未達到手臂側舉動作標準，請將手臂盡量舉高舉直，平行頸部'
                    elif Degree < 0:
                        message = f'第{num}個動作手臂延展過頭囉!請用標準動作進行復健以免受傷'
                    elif Degree == 0:
                        message = f'第{num}次動作符合標準!'
                    message_store.append(message)

                elif mode in ['ShoulderBladeStretch']:
                    # 若 mode 為其他動作 (如 LateralRaise, ShoulderBladeStretch...)
                    # 依照邏輯改為取最大角度
                    AngleList = [calculate_angle(points[0], points[1], points[2]) for points in InputSequence]
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

                    if len(PathRecord) > 0:
                        ExampleTargetPoints = max(PathRecord)  # PathRecord 為陣列比較時會比對第一維
                        ExampleAngle = calculate_angle(ExampleTargetPoints[0], ExampleTargetPoints[1], ExampleTargetPoints[2])
                    else:
                        ExampleAngle = TargetAngle

                    Degree = abs(int(ExampleAngle - TargetAngle))
                    if Degree > 0 and Degree <= 10:
                        message = f'第{num}個動作不正確，請您將肩胛拉伸幅度再擴大一點'
                    elif Degree > 10 and Degree <= 30:
                        message = f'第{num}個動作不正確，您需要再擴大肩胛拉伸幅度'
                    elif Degree > 30:
                        message = f'第{num}個動作並未達到肩胛拉伸要求，請擴大肩胛拉伸幅度'
                    elif Degree < 0:
                        message = f'第{num}個動作拉伸幅度太大囉!請用標準動作進行復健以免受傷'
                    elif Degree == 0:
                        message = f'第{num}次動作符合標準!'
                    message_store.append(message)
            else:
                # 若距離 <= threshold，表示相似度夠高，動作成功
                status = True
                message_store.append(f'第{num}次動作正確，符合標準!')

            message = ''
            for i, m in enumerate(message_store):
                if i >= 1:
                    m = '/'+m
                message+=m
    else:
        status = False
        message = '動作不正確，請您做出指定的動作><'
    print(message)
    return message, status

