using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement; // 引入場景管理

public class GameManager : MonoBehaviour
{
    private UDPReceive udpReceive; // 引用 UDPReceive
    public int final_score = 0;    // 加回 final_score，避免找不到錯誤



    // 新增 ToStart 方法，避免其他腳本報錯
    public void ToStart()
    {
        Debug.Log("場景切換至 Start Scene");
        SceneManager.LoadScene("Main Scene"); // 切換至 Start 場景
    }
    public void ToArm()
    {
        // Debug.Log("場景切換至 Start Scene");
        SceneManager.LoadScene("Arm"); // 切換至 Start 場景
    }
    public void ToSquat()
    {
        // Debug.Log("場景切換至 Start Scene");
        SceneManager.LoadScene("Squat"); // 切換至 Start 場景
    }    
    public void ToKnee()
    {
        // Debug.Log("場景切換至 Start Scene");
        SceneManager.LoadScene("Knee"); // 切換至 Start 場景
    }
    public void ToShoulder()
    {
        // Debug.Log("場景切換至 Start Scene");
        SceneManager.LoadScene("Shoulder"); // 切換至 Start 場景
    }

    public void ToMenuScene()
    {
        SceneManager.LoadScene("Menu");
    }
    public void ToScore()
    {
        // 將最終分數保存並切換到 Score 場景
        PlayerPrefs.SetInt("Score", final_score);
        Debug.Log("切換至 Score 場景，保存分數: " + final_score);
        SceneManager.LoadScene("Score");
    }
}
