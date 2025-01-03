# RestARt-重回健康人生
## 這裡會簡單說明每個程式碼檔案的功能

1. Detect.py
- 利用Mediapipe的cvzone套件去設計的python偵測系統，並沒有連接Unity去做動作傳輸的部分
2. Detect4.py
- 這是有利用socket去做傳輸Unity的動作傳輸
3. Train,py
- 這是用來訓練KNN模型的地方
4. DataCollect.py
- 收集復健動作的程式碼
