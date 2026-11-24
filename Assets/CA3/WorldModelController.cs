using System;
using UnityEngine;


[DefaultExecutionOrder(-5)]
public class WorldModelController : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public TextAsset controllerJson;

    // Must match ACTION_SMOOTH / MAX_DELTA in train_controller.py: the policy was optimised with
    // exactly this smoothing in imagination. Weight on the new raw action each step.
    [Range(0.0f, 1f)]
    public float actionSmoothing = 0.2f;

    public float maxDeltaPerStep = 0.05f;

    float[] W;
    float[] b;
    float[] jointLo;
    float[] jointHi;

    float[] features;
    float[] rawAction;
    float[] action;
    float[] state;
    bool ready;
    bool hasAction;
    bool tookOverEmitActions;
    bool previousEmitActions;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (goals == null)
            goals = GetComponent<ArmJointController>();
        if (agent == null || goals == null || controllerJson == null)
        {
            Debug.LogError("WorldModelController needs agent, goals, and controller.json.");
            enabled = false;
            return;
        }

        features = new float[agent.StateSize];
        rawAction = new float[agent.ActionSize];
        action = new float[agent.ActionSize];
        state = new float[agent.StateSize];

        if (!Load(controllerJson.text))
        {
            enabled = false;
            return;
        }

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        ready = true;
        Debug.Log(
            "WorldModelController ready (CMA-ES Controller C). " +
            "Goals come from ArmJointController; actions come from the learned linear policy.");
    }

    void OnEnable()
    {
        if (!ready || goals == null)
            return;

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        hasAction = false;
    }

    void OnDisable()
    {
        if (tookOverEmitActions && goals != null)
        {
            goals.emitActions = previousEmitActions;
            tookOverEmitActions = false;
        }
        hasAction = false;
    }

    unsafe void FixedUpdate()
    {
        if (!ready || !agent.Ready || goals.CurrentTarget == null ||
            !Mujoco.MjScene.InstanceExists || Mujoco.MjScene.Instance.Data == null)
            return;

        agent.CopyState(state);
        ActionFrom(state, rawAction);
        ApplySmoothing(rawAction, action);
        agent.SetAction(action);
    }

    // Policy action toward the ArmJointController's current goal (random goal or IK solution).
    public void ActionFrom(float[] state, float[] destination)
    {
        ActionFrom(state, goals.CurrentTarget, destination);
    }

    // Same linear policy as policy() in train_controller.py:
    //   features = [q - goal, qd],  action = goal + W * features + b
    public void ActionFrom(float[] state, float[] goal, float[] destination)
    {
        int joints = agent.ActionSize;
        int featureSize = agent.StateSize;
        for (int i = 0; i < joints; i++)
        {
            features[i] = state[i] - goal[i];
            features[joints + i] = state[joints + i];
        }

        for (int i = 0; i < joints; i++)
        {
            float u = b[i];
            int row = i * featureSize;
            for (int f = 0; f < featureSize; f++)
                u += W[row + f] * features[f];
            destination[i] = Mathf.Clamp(goal[i] + u, jointLo[i], jointHi[i]);
        }
    }

    // Stateless version for imagined rollouts (WorldModelDriver forecast): policy action from
    // `state`, smoothed relative to `previous` exactly like the live controller.
    public void ImagineAction(float[] state, float[] previous, float[] destination)
    {
        ActionFrom(state, destination);
        Smooth(previous, destination, destination);
    }

    void ApplySmoothing(float[] raw, float[] smoothed)
    {
        // First step after enabling: start from the current pose so the command does not jump.
        if (!hasAction)
        {
            for (int i = 0; i < agent.ActionSize; i++)
                action[i] = state[i];
            hasAction = true;
        }
        Smooth(action, raw, smoothed);
    }

    // smooth_action() in train_controller.py: low-pass toward the raw action, then rate-limit.
    void Smooth(float[] previous, float[] raw, float[] smoothed)
    {
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float prev = previous[i];
            float blended = (1f - actionSmoothing) * prev + actionSmoothing * raw[i];
            float delta = Mathf.Clamp(blended - prev, -maxDeltaPerStep, maxDeltaPerStep);
            smoothed[i] = Mathf.Clamp(prev + delta, jointLo[i], jointHi[i]);
        }
    }

    bool Load(string json)
    {
        try
        {
            var data = JsonUtility.FromJson<ControllerFile>(json);
            if (data == null || data.W == null || data.b == null ||
                data.W.Length != agent.ActionSize * agent.StateSize ||
                data.b.Length != agent.ActionSize)
            {
                Debug.LogError(
                    "controller.json has unexpected W/b shapes for ActionSize=" +
                    agent.ActionSize + " StateSize=" + agent.StateSize + ".");
                return false;
            }

            W = data.W;
            b = data.b;
            jointLo = data.joint_lo != null && data.joint_lo.Length == agent.ActionSize
                ? data.joint_lo
                : CopyRangesLo();
            jointHi = data.joint_hi != null && data.joint_hi.Length == agent.ActionSize
                ? data.joint_hi
                : CopyRangesHi();

            if (data.action_smooth > 0f && !Mathf.Approximately(data.action_smooth, actionSmoothing))
                Debug.LogWarning(
                    $"WorldModelController: actionSmoothing is {actionSmoothing} but the policy was trained with " +
                    $"{data.action_smooth}. Behaviour will differ from the imagined rollouts.", this);
            if (data.max_delta > 0f && !Mathf.Approximately(data.max_delta, maxDeltaPerStep))
                Debug.LogWarning(
                    $"WorldModelController: maxDeltaPerStep is {maxDeltaPerStep} but the policy was trained with " +
                    $"{data.max_delta}.", this);
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            return false;
        }
    }

    float[] CopyRangesLo()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].x
                : -Mathf.PI;
        return result;
    }

    float[] CopyRangesHi()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].y
                : Mathf.PI;
        return result;
    }

    [Serializable]
    class ControllerFile
    {
        public string feature;
        public int state_size;
        public int action_size;
        public float[] W;
        public float[] b;
        public float[] joint_lo;
        public float[] joint_hi;
        public float[] home;
        public float action_smooth = -1f;
        public float max_delta = -1f;
    }
}
