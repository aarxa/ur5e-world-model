using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using UnityEngine;

// Logs (s_t, a_t, s_{t+1}) every physics step while ArmJointController drives random joint goals.
// One CSV row = q (J) | qd (J) | q* (J, the commanded action) | q' (J) | qd' (J).
// Runs after ArmAgent has written this tick's ctrl and before MjScene steps, so s and a pair up.
[DefaultExecutionOrder(-1)]
public class RolloutRecorder : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController controller;
    public int steps = 70000;

    [Tooltip("Output path relative to the Assets folder. train.py reads Assets/CA3/Data/rollout.csv.")]
    public string outputFile = "CA3/Data/rollout.csv";

    [Tooltip("Speeds up collection. Physics still advances in fixed steps, so the data is unchanged.")]
    [Range(1f, 20f)]
    public float timeScale = 4f;

    public int logEvery = 5000;
    public bool stopPlayModeWhenDone = true;

    readonly List<float[]> states = new();
    readonly List<float[]> actions = new();
    readonly List<float[]> nextStates = new();

    int n;
    float[] prevS;
    float[] prevA;
    bool saved;
    float previousTimeScale = 1f;
    bool previousRunInBackground;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (controller == null)
            controller = GetComponent<ArmJointController>();
        if (agent == null || !agent.Ready)
        {
            Debug.LogError("RolloutRecorder needs a ready ArmAgent.", this);
            enabled = false;
            return;
        }
        if (controller == null || !controller.emitActions || !controller.randomGoals)
            Debug.LogWarning(
                "RolloutRecorder: collect with ArmJointController emitActions = true and randomGoals = true, " +
                "and with WorldModelController / IkGoalProvider disabled.", this);

        states.Capacity = steps;
        actions.Capacity = steps;
        nextStates.Capacity = steps;
        previousTimeScale = Time.timeScale;
        Time.timeScale = timeScale;
        // Keep simulating while Unity is not the focused app, so collection can run unattended.
        previousRunInBackground = Application.runInBackground;
        Application.runInBackground = true;
        Debug.Log($"RolloutRecorder: collecting {steps} steps at dt = {Time.fixedDeltaTime:0.####} s.", this);
    }

    unsafe void FixedUpdate()
    {
        if (saved || !agent.Ready || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;

        var s = (float[])agent.Observe().Clone();
        var a = (float[])agent.CurrentAction.Clone();

        if (prevS != null)
        {
            states.Add(prevS);
            actions.Add(prevA);
            nextStates.Add(s);
            n++;
            if (logEvery > 0 && n % logEvery == 0)
                Debug.Log($"RolloutRecorder: {n}/{steps} steps, joint tracking RMS {TrackingError():0.000} rad.", this);
        }
        prevS = s;
        prevA = a;

        if (n >= steps)
        {
            Save();
            saved = true;
            Time.timeScale = previousTimeScale;
#if UNITY_EDITOR
            if (stopPlayModeWhenDone)
                UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }

    void OnDisable()
    {
        // Keep a partial rollout if Play mode is stopped early.
        if (!saved && n > 0)
        {
            Debug.LogWarning($"RolloutRecorder stopped early; saving {n} steps.", this);
            Save(refreshAssets: false);
            saved = true;
        }
        Time.timeScale = previousTimeScale;
        Application.runInBackground = previousRunInBackground;
    }

    float TrackingError()
    {
        var goal = controller != null ? controller.CurrentTarget : agent.CurrentAction;
        var s = agent.Observe();
        if (goal == null || s == null)
            return 0f;
        float sum = 0f;
        for (int i = 0; i < agent.JointCount; i++)
        {
            float e = s[i] - goal[i];
            sum += e * e;
        }
        return Mathf.Sqrt(sum / Mathf.Max(1, agent.JointCount));
    }

    void Save(bool refreshAssets = true)
    {
        int j = agent.JointCount;
        var culture = CultureInfo.InvariantCulture;
        var sb = new StringBuilder(n * (2 * agent.StateSize + agent.ActionSize) * 12);

        // np.loadtxt skips lines starting with '#', so the header documents columns without
        // breaking train.py.
        sb.Append('#');
        AppendNames(sb, "q", j, first: true);
        AppendNames(sb, "qd", j);
        AppendNames(sb, "qstar", j);
        AppendNames(sb, "qnext", j);
        AppendNames(sb, "qdnext", j);
        sb.Append('\n');

        for (int row = 0; row < n; row++)
        {
            AppendRow(sb, states[row], culture, first: true);
            AppendRow(sb, actions[row], culture);
            AppendRow(sb, nextStates[row], culture);
            sb.Append('\n');
        }

        var path = Path.Combine(Application.dataPath, outputFile);
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"RolloutRecorder: wrote {n} transitions to {path}", this);
#if UNITY_EDITOR
        if (refreshAssets)
            UnityEditor.AssetDatabase.Refresh();
#endif
    }

    static void AppendNames(StringBuilder sb, string prefix, int count, bool first = false)
    {
        for (int i = 0; i < count; i++)
        {
            if (!first || i > 0)
                sb.Append(',');
            sb.Append(prefix).Append(i);
        }
    }

    static void AppendRow(StringBuilder sb, float[] values, CultureInfo culture, bool first = false)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!first || i > 0)
                sb.Append(',');
            sb.Append(values[i].ToString("R", culture));
        }
    }
}
