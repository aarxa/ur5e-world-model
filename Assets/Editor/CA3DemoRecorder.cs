using System.IO;
using UnityEditor;
using UnityEditor.Recorder;
using UnityEditor.Recorder.Encoder;
using UnityEditor.Recorder.Input;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// Records a short MP4 of Step 4 (IK tracking) with Unity Recorder.
// Menu: CA3 > Record IK Demo Video. Output: Recordings/ur5e_ik_demo.mp4 in the project folder.
// It enters Play mode, frames the arm with the Main Camera, lets the arm settle on the target,
// then moves TrackingTarget twice around a loop in front of the arm while recording the Game view.
// Camera and target changes happen in Play mode only, so the saved scene is untouched.
[InitializeOnLoad]
public static class CA3DemoRecorder
{
    const string ScenePath = "Assets/CA3/CA3.unity";
    const string PendingKey = "CA3DemoRecorder.Pending";
    const string QuitKey = "CA3DemoRecorder.Quit";
    const float SettleSeconds = 2.5f;   // arm moves from its start pose onto the target
    const float PathSeconds = 24f;      // two loops around the path
    const float HoldSeconds = 2f;       // back at the start point

    static RecorderController controller;
    static Transform target;
    static Vector3 start;

    static CA3DemoRecorder()
    {
        EditorApplication.playModeStateChanged += OnPlayModeChanged;
    }

    [MenuItem("CA3/Record IK Demo Video")]
    static void RecordFromMenu() => Begin(false);

    // For unity -executeMethod CA3DemoRecorder.RecordAndQuit
    public static void RecordAndQuit() => Begin(true);

    static void Begin(bool quitWhenDone)
    {
        if (EditorApplication.isPlaying)
        {
            Debug.LogError("CA3 demo: stop Play mode first.");
            return;
        }
        if (SceneManager.GetActiveScene().path != ScenePath)
            EditorSceneManager.OpenScene(ScenePath);
        if (GameObject.Find("TrackingTarget") == null || Object.FindFirstObjectByType<IkGoalProvider>() == null)
        {
            Debug.LogError("CA3 demo: run CA3 > Set Up IK Tracking (Step 4) first.");
            if (quitWhenDone)
                EditorApplication.Exit(1);
            return;
        }

        // Entering Play mode reloads scripts, so remember the request across the reload.
        SessionState.SetBool(PendingKey, true);
        SessionState.SetBool(QuitKey, quitWhenDone);
        EditorApplication.isPlaying = true;
    }

    static void OnPlayModeChanged(PlayModeStateChange state)
    {
        if (!SessionState.GetBool(PendingKey, false))
            return;

        if (state == PlayModeStateChange.EnteredPlayMode)
        {
            StartDemo();
        }
        else if (state == PlayModeStateChange.EnteredEditMode)
        {
            bool quit = SessionState.GetBool(QuitKey, false);
            SessionState.EraseBool(PendingKey);
            SessionState.EraseBool(QuitKey);
            if (quit)
                EditorApplication.Exit(0);
        }
    }

    static void StartDemo()
    {
        target = GameObject.Find("TrackingTarget").transform;
        start = target.position;

        // Slightly larger target so it shows next to the tool (Play mode only).
        target.localScale = Vector3.one * 0.065f;

        var cam = Camera.main;
        if (cam != null)
        {
            // Centre of the arm plus the target's path, viewed from the front-right at about 1 m.
            var look = start + new Vector3(0.081f, -0.058f, -0.278f);
            cam.transform.position = look + new Vector3(1.0f, 0.5f, 1.15f).normalized * 1.35f;
            cam.transform.LookAt(look);
            cam.fieldOfView = 38f;
        }

        string output = Path.Combine(Path.GetDirectoryName(Application.dataPath), "Recordings", "ur5e_ik_demo");
        Directory.CreateDirectory(Path.GetDirectoryName(output));

        var movie = ScriptableObject.CreateInstance<MovieRecorderSettings>();
        movie.name = "CA3 IK demo";
        movie.Enabled = true;
        movie.EncoderSettings = new CoreEncoderSettings
        {
            Codec = CoreEncoderSettings.OutputCodec.MP4,
            EncodingQuality = CoreEncoderSettings.VideoEncodingQuality.High,
        };
        movie.ImageInputSettings = new GameViewInputSettings { OutputWidth = 1280, OutputHeight = 720 };
        movie.AudioInputSettings.PreserveAudio = false;
        movie.OutputFile = output;

        var settings = ScriptableObject.CreateInstance<RecorderControllerSettings>();
        settings.AddRecorderSettings(movie);
        settings.SetRecordModeToManual();
        settings.FrameRate = 30f;

        controller = new RecorderController(settings);
        controller.PrepareRecording();
        controller.StartRecording();
        EditorApplication.update += Tick;
        Debug.Log($"CA3 demo: recording {SettleSeconds + PathSeconds + HoldSeconds:0} s to {output}.mp4");
    }

    static void Tick()
    {
        if (!EditorApplication.isPlaying || target == null)
        {
            EditorApplication.update -= Tick;
            return;
        }

        // Recorder plays back at a constant 30 fps, so game time (not wall-clock time) drives the path.
        float t = Time.time;
        target.position = start + PathOffset(t);

        if (t >= SettleSeconds + PathSeconds + HoldSeconds)
        {
            EditorApplication.update -= Tick;
            controller.StopRecording();
            Debug.Log("CA3 demo: recording finished.");
            EditorApplication.isPlaying = false;
        }
    }

    // A 0.15 m-radius loop around the tool's home position plus a 0.1 m up-and-down figure-eight.
    // The same path (in MuJoCo coordinates) stayed within 2 mm in the offline IK test.
    static Vector3 PathOffset(float t)
    {
        if (t < SettleSeconds)
            return Vector3.zero;
        float s = Mathf.Min(t - SettleSeconds, PathSeconds);
        float phase = s / PathSeconds * 4f * Mathf.PI;
        float mjX = 0.15f * (Mathf.Cos(phase) - 1f);
        float mjY = 0.15f * Mathf.Sin(phase);
        float mjZ = 0.10f * Mathf.Sin(2f * phase);   // MuJoCo z is up
        return new Vector3(mjX, mjZ, mjY);           // Unity (x, up, z)
    }
}
