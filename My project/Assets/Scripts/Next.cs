using UnityEngine;
using UnityEngine.SceneManagement;
namespace NextScene
{
    public class SceneController : MonoBehaviour
    {
        // 切換到下一個場景（依 Build Settings 中的順序）
        public void LoadNextScene()
        {
            int currentSceneIndex = SceneManager.GetActiveScene().buildIndex;
            SceneManager.LoadScene(currentSceneIndex + 1);
            }
    }
}