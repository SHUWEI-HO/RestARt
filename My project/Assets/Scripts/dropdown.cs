using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class DropdownController : MonoBehaviour
{
    private GameManager gM;
    public Dropdown dropdown;

    void Start()
    {
        // 設定下拉選單的事件
        dropdown.onValueChanged.AddListener(OnDropdownValueChanged);
    }

    // 設置為 public，使其可以在 Inspector 中選擇
    public void OnDropdownValueChanged(int index)
    {
        switch (index)
        {
            case 1: // 開始
                SceneManager.LoadScene("Main");
                break;
            case 0: // 復健動作
                SceneManager.LoadScene("Menu"); // 或是具體的復健動作場景
                break;
            // case 2: // 設定
            //     SceneManager.LoadScene("Settings");
            //     break;
        }
    }
}
