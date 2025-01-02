using UnityEngine;
using UnityEngine.UI;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class TimeReceiver : MonoBehaviour
{
    public int port = 8106;
    public Text errorText;

    private Thread receiveThread;
    private UdpClient udpClient;
    private string latestMessage = "";
    private readonly object messageLock = new object();

    void Start()
    {
        udpClient = new UdpClient(port);
        receiveThread = new Thread(ReceiveMessages);
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private async void ReceiveMessages()
    {
        while (true)
        {
            try
            {
                UdpReceiveResult result = await udpClient.ReceiveAsync();
                string receivedMessage = Encoding.UTF8.GetString(result.Buffer);

                // 將接收到的數字轉換為時間格式
                string formattedMessage = FormatAsTime(receivedMessage);

                lock (messageLock)
                {
                    latestMessage = formattedMessage;
                }

                Debug.Log("[SocketReceiver] Received: " + formattedMessage);
            }
            catch (Exception e)
            {
                Debug.LogError("[SocketReceiver] Error: " + e.Message);
                break;
            }
        }
    }

    // 將數字轉換為時間格式的方法
    private string FormatAsTime(string message)
    {
        if (int.TryParse(message, out int seconds))
        {
            // 計算分鐘和秒數
            int minutes = seconds / 60;
            int remainingSeconds = seconds % 60;

            // 格式化為 "mm:ss"
            return string.Format("{0:D2}:{1:D2}", minutes, remainingSeconds);
        }
        else
        {
            Debug.LogWarning("[SocketReceiver] Invalid number format: " + message);
            return "Invalid Time";
        }
    }

    void Update()
    {
        lock (messageLock)
        {
            if (errorText != null && !string.IsNullOrEmpty(latestMessage))
            {
                errorText.text = latestMessage;
            }
        }
    }

    private void OnDestroy()
    {
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
        }

        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
}