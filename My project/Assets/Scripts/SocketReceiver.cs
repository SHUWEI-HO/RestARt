using UnityEngine;
using UnityEngine.UI;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class SocketReceiver : MonoBehaviour
{
    public int port = 7801;
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

                string formattedMessage = receivedMessage.Replace("/", "\r\n");
                

                lock (messageLock)
                {
                    latestMessage = formattedMessage ;
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