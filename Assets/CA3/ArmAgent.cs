using System;
using Mujoco;
using UnityEngine;

// Thin wrapper around the imported MuJoCo arm.
//   state  = [q_0..q_{J-1}, qd_0..qd_{J-1}]   (radians, radians/second)
//   action = one position target per actuator  (radians)
//
// Execution order inside one FixedUpdate tick:
//   IkGoalProvider (-20) -> ArmJointController (-10) -> WorldModelController (-5)
//   -> ArmAgent (-2, writes ctrl) -> RolloutRecorder (-1, logs s/a)
//   -> MjScene (0, mj_step) -> WorldModelDriver (10, compares prediction to the new state)
[DefaultExecutionOrder(-2)]
public class ArmAgent : MonoBehaviour
{
    public int JointCount;
    public int StateSize;
    public int ActionSize;

    public MjActuator gripper;
    public MjBody endEffectorBody;
    public Transform endEffector;
    public MjHingeJoint[] joints;
    public MjActuator[] actuators;

    [Header("MJCF names (UR5e from MuJoCo Menagerie)")]
    public string[] jointNames =
    {
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    };
    public string[] actuatorNames =
    {
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3",
    };
    public string endEffectorBodyName = "wrist_3_link";
    public string endEffectorSiteName = "attachment_site";

    [Tooltip("Tool flange site. When set, it is the point the IK solver, ghost marker and deviation line use.")]
    public MjSite endEffectorSite;

    [Tooltip("Constant command for the gripper actuator, if the arm has one (the UR5e does not).")]
    public float gripperCommand;

    float[] action;
    float[] state;
    bool ready;
    bool hasAction;

    public float[] CurrentAction => action;
    public bool Ready => ready;

    void Awake()
    {
        InitializeJointsAndActuators();
    }

    void FixedUpdate()
    {
        if (!ready)
            return;
        ApplyAction();
    }

    public void InitializeJointsAndActuators()
    {
        ready = false;

        if (joints == null || joints.Length == 0 || Array.Exists(joints, j => j == null))
        {
            joints = new MjHingeJoint[jointNames.Length];
            for (int i = 0; i < jointNames.Length; i++)
                joints[i] = FindComponent<MjHingeJoint>(jointNames[i]);
        }

        if (actuators == null || actuators.Length != joints.Length || Array.Exists(actuators, a => a == null))
        {
            // Pair each actuator with its joint through MjActuator.Joint so the action order always
            // matches the state order, then fall back to the MJCF actuator names.
            var all = GetComponentsInChildren<MjActuator>(true);
            if (all.Length == 0)
                all = FindObjectsByType<MjActuator>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            actuators = new MjActuator[joints.Length];
            for (int i = 0; i < joints.Length; i++)
            {
                if (joints[i] != null)
                    actuators[i] = Array.Find(all, a => a != gripper && a.Joint == joints[i]);
                if (actuators[i] == null && i < actuatorNames.Length)
                    actuators[i] = FindComponent<MjActuator>(actuatorNames[i]);
            }
        }

        if (endEffectorBody == null)
            endEffectorBody = FindComponent<MjBody>(endEffectorBodyName);
        if (endEffectorSite == null && !string.IsNullOrEmpty(endEffectorSiteName))
            endEffectorSite = FindComponent<MjSite>(endEffectorSiteName);
        if (endEffector == null)
            endEffector = endEffectorSite != null ? endEffectorSite.transform
                : endEffectorBody != null ? endEffectorBody.transform : null;

        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] == null)
            {
                Debug.LogError($"ArmAgent: joint '{Name(jointNames, i)}' not found under {name}.", this);
                return;
            }
            if (actuators[i] == null)
            {
                Debug.LogError($"ArmAgent: no actuator drives joint '{joints[i].name}'.", this);
                return;
            }
        }
        if (endEffectorBody == null)
        {
            Debug.LogError($"ArmAgent: end-effector body '{endEffectorBodyName}' not found.", this);
            return;
        }

        JointCount = joints.Length;
        ActionSize = actuators.Length;
        StateSize = 2 * JointCount;
        action = new float[ActionSize];
        state = new float[StateSize];
        hasAction = false;
        ready = true;
    }

    public void SetAction(float[] targets)
    {
        if (!ready || targets == null)
            return;
        int count = Mathf.Min(targets.Length, ActionSize);
        for (int i = 0; i < count; i++)
            action[i] = targets[i];
        hasAction = true;
    }

    public unsafe void ApplyAction()
    {
        if (!ready || !MjScene.InstanceExists)
            return;
        var data = MjScene.Instance.Data;
        if (data == null)
            return;

        // Until something sets an action, hold the current pose instead of driving to zeros.
        if (!hasAction)
        {
            for (int i = 0; i < ActionSize; i++)
                action[i] = (float)data->qpos[joints[i].QposAddress];
            hasAction = true;
        }

        for (int i = 0; i < ActionSize; i++)
        {
            // MjActuator.Control is only copied to MuJoCo after mj_step (MjScene.SyncUnityToMjState),
            // so also write ctrl directly. This makes the action set this tick drive this tick's step.
            actuators[i].Control = action[i];
            data->ctrl[actuators[i].MujocoId] = action[i];
        }

        if (gripper != null)
        {
            gripper.Control = gripperCommand;
            data->ctrl[gripper.MujocoId] = gripperCommand;
        }
    }

    public unsafe float[] Observe()
    {
        if (!ready || !MjScene.InstanceExists)
            return state;
        var data = MjScene.Instance.Data;
        if (data == null)
            return state;

        for (int i = 0; i < JointCount; i++)
        {
            state[i] = (float)data->qpos[joints[i].QposAddress];
            state[JointCount + i] = (float)data->qvel[joints[i].DofAddress];
        }
        return state;
    }

    public void CopyState(float[] destination)
    {
        var current = Observe();
        if (destination == null || current == null)
            return;
        Array.Copy(current, destination, Mathf.Min(destination.Length, current.Length));
    }

    // End-effector point in Unity world coordinates for whatever qpos `data` currently holds
    // (callers run mj_kinematics first). Uses the tool site when present, else the body origin.
    public unsafe Vector3 EndEffectorPosition(MujocoLib.mjData_* data)
    {
        if (endEffectorSite != null && endEffectorSite.MujocoId >= 0)
            return MjEngineTool.UnityVector3(MjEngineTool.MjVector3AtEntry(data->site_xpos, endEffectorSite.MujocoId));
        return MjEngineTool.UnityVector3(MjEngineTool.MjVector3AtEntry(data->xpos, endEffectorBody.MujocoId));
    }

    private T FindComponent<T>(string name) where T : Component
    {
        if (string.IsNullOrEmpty(name))
            return null;

        // Prefer the arm this agent sits on, so a duplicated ghost arm in the scene is never picked.
        foreach (var component in GetComponentsInChildren<T>(true))
            if (component.gameObject.name == name)
                return component;

        foreach (var component in FindObjectsByType<T>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            if (component.gameObject.name == name)
            {
                Debug.LogWarning($"ArmAgent: '{name}' found outside {this.name}; put ArmAgent on the arm root.", this);
                return component;
            }

        return null;
    }

    static string Name(string[] names, int i) => names != null && i < names.Length ? names[i] : $"#{i}";
}
