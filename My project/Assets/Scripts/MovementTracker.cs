using UnityEngine;
using System.Collections.Generic;

public class MovementTracker : MonoBehaviour
{
    public Transform coachTransform; // 健身教練的 Transform（或骨骼/關節物件）
    public Transform userTransform;  // 使用者的 Transform（通過姿勢偵測追蹤）
    public float threshold = 0.1f;   // 動作匹配的閾值
    public float matchTime = 1.0f;   // 需要匹配多長時間才算成功
    private float matchTimer = 0f;   // 用來追蹤匹配成功的時間
    public int successCount = 0;     // 成功的次數

    // 要比對的關節（此範例簡化為手和膝蓋）
    public Transform[] coachJoints;
    public Transform[] userJoints;

    // 每幀都會調用一次 Update
    void Update()
    {
        if (CompareMovement())
        {
            matchTimer += Time.deltaTime;
            if (matchTimer >= matchTime)
            {
                Success();
                matchTimer = 0f; // 成功後重置計時器
            }
        }
        else
        {
            matchTimer = 0f; // 如果動作不匹配，重置計時器
        }
    }

    // 比對動作的函數
    bool CompareMovement()
    {
        for (int i = 0; i < coachJoints.Length; i++)
        {
            float distance = Vector3.Distance(coachJoints[i].position, userJoints[i].position);
            if (distance > threshold)
            {
                return false; // 動作沒有在閾值內匹配
            }
        }
        return true; // 所有關節都在閾值內匹配
    }

    // 成功時的處理函數
    void Success()
    {
        successCount++;
        Debug.Log("成功次數: " + successCount);
        // 可以加入 UI 或回饋通知使用者已成功
    }
}
