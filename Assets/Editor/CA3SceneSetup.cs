using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

// Editor helper: Unity menu "CA3 > Make Ghost Prefab + Materials".
// Saves the open scene as Assets/CA3/CA3.unity, copies the imported arm into a visual-only ghost
// prefab (Assets/CA3/ur5e_ghost.prefab) and creates the Ghost and Line materials that
// WorldModelDriver uses. Existing files are left alone.
public static class CA3SceneSetup
{
    const string Folder = "Assets/CA3";

    [MenuItem("CA3/Make Ghost Prefab + Materials")]
    static void Make()
    {
        var arm = Selection.activeGameObject != null ? Selection.activeGameObject : GameObject.Find("ur5e");
        if (arm == null)
        {
            EditorUtility.DisplayDialog("CA3", "Import ur5e.xml first (Assets > Import MuJoCo Scene), then select the ur5e object.", "OK");
            return;
        }
        if (arm.GetComponent("ArmAgent") != null)
        {
            EditorUtility.DisplayDialog("CA3", $"'{arm.name}' already has CA3 scripts. The ghost must be copied from a clean import.", "OK");
            return;
        }

        if (PrefabUtility.IsPartOfPrefabInstance(arm))
            PrefabUtility.UnpackPrefabInstance(arm, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);

        var scene = arm.scene;
        if (string.IsNullOrEmpty(scene.path))
            EditorSceneManager.SaveScene(scene, Folder + "/CA3.unity");

        // SaveAsPrefabAsset (not ...AndConnect) leaves the scene arm a plain object, so scripts added
        // to it later never end up in the ghost.
        string prefabPath = Folder + "/ur5e_ghost.prefab";
        if (AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
            PrefabUtility.SaveAsPrefabAsset(arm, prefabPath);

        if (AssetDatabase.LoadAssetAtPath<Material>(Folder + "/Ghost.mat") == null)
        {
            var ghost = new Material(Shader.Find("Standard")) { name = "Ghost" };
            // Standard shader "Fade" rendering mode.
            ghost.SetFloat("_Mode", 2f);
            ghost.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
            ghost.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
            ghost.SetInt("_ZWrite", 0);
            ghost.DisableKeyword("_ALPHATEST_ON");
            ghost.EnableKeyword("_ALPHABLEND_ON");
            ghost.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            ghost.renderQueue = (int)RenderQueue.Transparent;
            ghost.color = new Color(0.45f, 0.75f, 1f, 0.3f);
            AssetDatabase.CreateAsset(ghost, Folder + "/Ghost.mat");
        }

        if (AssetDatabase.LoadAssetAtPath<Material>(Folder + "/Line.mat") == null)
        {
            var line = new Material(Shader.Find("Sprites/Default")) { name = "Line", color = Color.yellow };
            AssetDatabase.CreateAsset(line, Folder + "/Line.mat");
        }

        AssetDatabase.SaveAssets();
        EditorSceneManager.SaveScene(scene);
        Selection.activeObject = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
        Debug.Log("CA3: saved Assets/CA3/CA3.unity and created ur5e_ghost.prefab, Ghost.mat and Line.mat.");
    }

    // Step 2 "Collect rollout" mode from SCENE_MODES.txt: agent + random goals + recorder on the arm,
    // later-stage components switched off.
    [MenuItem("CA3/Set Up Rollout Recording")]
    static void SetUpRecording()
    {
        // The importer names the root after the file plus a number (e.g. "ur5e343"), so find the arm
        // through one of its joints instead of by name.
        var joint = GameObject.Find("shoulder_pan_joint");
        var arm = joint != null ? joint.transform.root.gameObject : null;
        if (arm == null)
        {
            EditorUtility.DisplayDialog("CA3", "No imported UR5e (shoulder_pan_joint) in the open scene. Open Assets/CA3/CA3.unity first.", "OK");
            return;
        }

        GetOrAdd<ArmAgent>(arm);
        var goals = GetOrAdd<ArmJointController>(arm);
        goals.emitActions = true;
        goals.randomGoals = true;
        GetOrAdd<RolloutRecorder>(arm).enabled = true;

        foreach (var c in arm.GetComponents<WorldModelController>()) c.enabled = false;
        foreach (var c in arm.GetComponents<WorldModelDriver>()) c.enabled = false;
        foreach (var c in arm.GetComponents<IkGoalProvider>()) c.enabled = false;

        EditorSceneManager.MarkSceneDirty(arm.scene);
        EditorSceneManager.SaveScene(arm.scene);
        Selection.activeGameObject = arm;
        Debug.Log("CA3: ur5e is set up for rollout recording. Press Play and wait for 'RolloutRecorder: wrote'.");
    }

    // Step 3: learned controller drives the arm toward the same random goals used for data
    // collection, and WorldModelDriver shows the world model's ghost, forecast and deviation lines.
    [MenuItem("CA3/Set Up Learned Controller + Ghost (Step 3)")]
    static void SetUpController()
    {
        var arm = FindArm();
        if (arm == null || !WireController(arm))
            return;
        foreach (var c in arm.GetComponents<IkGoalProvider>()) c.enabled = false;
        arm.GetComponent<ArmJointController>().randomGoals = true;
        Save(arm, "CA3: learned controller + ghost set up on random goals. Press Play.");
    }

    // Step 4: a collider-free TrackingTarget sphere at the tool flange's home position, and IK that
    // turns its position into joint goals for the learned controller.
    [MenuItem("CA3/Set Up IK Tracking (Step 4)")]
    static void SetUpIk()
    {
        var arm = FindArm();
        if (arm == null || !WireController(arm))
            return;

        var target = GameObject.Find("TrackingTarget");
        if (target == null)
        {
            target = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            target.name = "TrackingTarget";
            Object.DestroyImmediate(target.GetComponent<Collider>());
            target.transform.localScale = Vector3.one * 0.05f;
            // attachment_site at the UR5e home pose, in Unity coordinates relative to the arm root.
            target.transform.position = arm.transform.TransformPoint(new Vector3(-0.134f, 0.488f, 0.492f));
            var line = AssetDatabase.LoadAssetAtPath<Material>(Folder + "/Line.mat");
            if (line != null)
                target.GetComponent<MeshRenderer>().sharedMaterial = line;
            Undo.RegisterCreatedObjectUndo(target, "Create TrackingTarget");
        }

        var ik = GetOrAdd<IkGoalProvider>(arm);
        ik.target = target.transform;
        ik.enabled = true;
        Selection.activeGameObject = target;
        Save(arm, "CA3: IK tracking set up. Press Play, then drag TrackingTarget in the Scene view.");
    }

    static GameObject FindArm()
    {
        var joint = GameObject.Find("shoulder_pan_joint");
        var arm = joint != null ? joint.transform.root.gameObject : null;
        if (arm == null)
            EditorUtility.DisplayDialog("CA3", "No imported UR5e (shoulder_pan_joint) in the open scene. Open Assets/CA3/CA3.unity first.", "OK");
        return arm;
    }

    static bool WireController(GameObject arm)
    {
        AssetDatabase.Refresh();
        var controllerJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Folder + "/Data/controller.json");
        var normJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Folder + "/Data/norm.json");
        var model = AssetDatabase.LoadMainAssetAtPath(Folder + "/Data/dynamics.onnx");
        var ghost = AssetDatabase.LoadAssetAtPath<GameObject>(Folder + "/ur5e_ghost.prefab");
        var ghostMat = AssetDatabase.LoadAssetAtPath<Material>(Folder + "/Ghost.mat");
        var lineMat = AssetDatabase.LoadAssetAtPath<Material>(Folder + "/Line.mat");

        string missing = "";
        if (controllerJson == null) missing += "\n Data/controller.json (run train_controller.py)";
        if (normJson == null) missing += "\n Data/norm.json (run train.py)";
        if (model == null) missing += "\n Data/dynamics.onnx (run train.py)";
        if (ghost == null || ghostMat == null || lineMat == null) missing += "\n ghost prefab / materials (CA3 > Make Ghost Prefab + Materials)";
        if (missing.Length > 0)
        {
            EditorUtility.DisplayDialog("CA3", "Missing in Assets/CA3:" + missing, "OK");
            return false;
        }

        GetOrAdd<ArmAgent>(arm);
        GetOrAdd<ArmJointController>(arm);
        foreach (var c in arm.GetComponents<RolloutRecorder>()) c.enabled = false;

        var policy = GetOrAdd<WorldModelController>(arm);
        policy.controllerJson = controllerJson;
        policy.enabled = true;

        var driver = GetOrAdd<WorldModelDriver>(arm);
        var so = new SerializedObject(driver);
        so.FindProperty("modelAsset").objectReferenceValue = model;
        so.FindProperty("normJson").objectReferenceValue = normJson;
        so.FindProperty("ghostArm").objectReferenceValue = ghost.transform;
        so.FindProperty("ghostMaterial").objectReferenceValue = ghostMat;
        so.FindProperty("lineMaterial").objectReferenceValue = lineMat;
        so.ApplyModifiedProperties();
        driver.enabled = true;
        return true;
    }

    static void Save(GameObject arm, string message)
    {
        EditorSceneManager.MarkSceneDirty(arm.scene);
        EditorSceneManager.SaveScene(arm.scene);
        Debug.Log(message);
    }

    static T GetOrAdd<T>(GameObject go) where T : Component
    {
        var component = go.GetComponent<T>();
        if (component == null)
            component = Undo.AddComponent<T>(go);
        return component;
    }
}

// Unity's Pause toggle stays on across Play sessions; with it on, Play runs one frame and freezes,
// which looks like the arm "not moving". Turn it off whenever Play starts paused, and for the first
// 10 s of each Play session log a heartbeat showing whether physics time is actually advancing.
[InitializeOnLoad]
static class CA3UnpauseOnPlay
{
    static double playStart = -1, nextBeat;
    static readonly bool logHeartbeat = false;   // set true to log pause/timeScale/physics time for 10 s of each Play

    static CA3UnpauseOnPlay()
    {
        EditorApplication.playModeStateChanged += state =>
        {
            if (state == PlayModeStateChange.EnteredPlayMode)
            {
                playStart = EditorApplication.timeSinceStartup;
                nextBeat = playStart + 1.0;
                // Keep simulating when Unity is not the focused app (Player Settings "Run In Background"
                // is off by default and also governs Play mode in the editor).
                Application.runInBackground = true;
                if (EditorApplication.isPaused)
                {
                    EditorApplication.isPaused = false;
                    Debug.Log("CA3: Unity's Pause button was on, so Play was frozen after one frame. Turned it off.");
                }
                if (Time.timeScale == 0f)
                {
                    Time.timeScale = 1f;
                    Debug.Log("CA3: Time.timeScale was 0, so physics never advanced. Set it to 1.");
                }
            }
            else if (state == PlayModeStateChange.ExitingPlayMode)
            {
                playStart = -1;
            }
        };

        EditorApplication.update += () =>
        {
            if (playStart < 0 || !EditorApplication.isPlaying)
                return;
            double now = EditorApplication.timeSinceStartup;
            if (now < nextBeat)
                return;
            nextBeat = now + 2.0;
            if (now - playStart > 10.0)
            {
                playStart = -1;
                return;
            }
            if (!logHeartbeat)
                return;
            Debug.Log($"CA3: heartbeat {now - playStart:0}s into Play: paused={EditorApplication.isPaused}, " +
                      $"timeScale={Time.timeScale}, physics time={Time.fixedTime:0.00}s, frames={Time.frameCount}");
        };
    }
}
