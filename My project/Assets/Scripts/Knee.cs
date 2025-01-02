using UnityEngine;
using UnityEngine.SceneManagement;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace knee
{
    public class SceneController : MonoBehaviour
    {
        public string pythonIP = "127.0.0.1";
        public int pythonPort = 5053;

        public void LoadNextScene()
        {
            // int currentSceneIndex = SceneManager.GetActiveScene().buildIndex;
            // SceneManager.LoadScene("Knee");

            // 改用 Socket，而非 UdpClient
            Socket sock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            IPEndPoint endPoint = new IPEndPoint(IPAddress.Parse(pythonIP), pythonPort);

            // 若 Encoding 也不可用，可以自行建 byte[]
            byte[] sendBytes = Encoding.UTF8.GetBytes("3, Going");

            sock.SendTo(sendBytes, endPoint);
            sock.Close();

            Debug.Log("[Unity->Python] 以原生 Socket 傳送動作編號: 3, Going");
        }
    }
}