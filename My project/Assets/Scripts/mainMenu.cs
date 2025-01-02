using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class mainMenu : MonoBehaviour
{
    void Start()
    {

    }

    void Update()
    {

    }

    public void playmain()
    {
        SceneManager.LoadScene("main");
    }

    public void playLevel2()
    {
        SceneManager.LoadScene("level2");
    }

    public void playLevel1()
    {
        SceneManager.LoadScene("level1");
    }

    public void exitGame()
    {
        Application.Quit();
    }
}