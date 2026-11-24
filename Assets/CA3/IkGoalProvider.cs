using Mujoco;
using UnityEngine;

// Cartesian IK that turns a movable target position into a joint-space goal for the learned
// controller.
//
// Method: damped least squares (Levenberg-Marquardt) on end-effector position.
//   e  = target - p(q)                                (3-vector, Unity world space)
//   J  = dp/dq by finite differences of MuJoCo's own forward kinematics (3 x J)
//   dq = J^T (J J^T + lambda^2 I)^-1 e  +  (I - J^+ J) * k (q_rest - q)
// The UR5e has 6 joints but a position target only fixes 3 numbers, so the null-space term pulls
// the spare degrees of freedom back toward the home posture. That keeps the wrist sensible and
// the solution inside the joint region the world model and controller were trained on.
// The solve warm-starts from last tick's answer, so a slowly moving target needs one or two
// iterations per physics step.
[DefaultExecutionOrder(-20)]
public class IkGoalProvider : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    [Tooltip("Collider-free GameObject the end effector should reach (e.g. a sphere named TrackingTarget).")]
    public Transform target;

    [Header("Damped least squares")]
    public int iterations = 30;
    [Tooltip("lambda: larger is steadier near singularities and unreachable targets, smaller is more exact.")]
    public float damping = 0.05f;
    [Tooltip("Largest joint change per iteration (radians).")]
    public float maxStep = 0.2f;
    [Tooltip("Stop iterating once the end effector is this close to the target (meters).")]
    public float tolerance = 0.002f;
    public float finiteDifferenceStep = 1e-4f;

    [Header("Posture")]
    [Tooltip("Null-space pull toward homeTargets, per iteration.")]
    public float nullSpaceGain = 0.05f;
    [Tooltip("Restrict solutions to homeTargets +/- the random-goal span the rollouts were collected over.")]
    public bool stayInTrainingRegion = true;

    [Header("Debug")]
    public bool drawGizmos = true;

    public float LastError { get; private set; }
    public float[] Solution => q;

    float[] q;
    float[] qRest;
    float[] lo;
    float[] hi;
    double[] savedQ;
    float[] dq;
    Vector3[] jacobian;
    bool ready;
    bool seeded;
    bool tookOverGoals;
    bool previousRandomGoals;
    Vector3 lastEndEffector;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (goals == null)
            goals = GetComponent<ArmJointController>();
        if (target == null)
        {
            var found = GameObject.Find("TrackingTarget");
            if (found != null)
                target = found.transform;
        }
        // goals.Ready is not checked: this Start runs before ArmJointController.Start (order -20 vs
        // -10). Only its serialized ranges/home are needed here; SetTarget is first called in
        // FixedUpdate, after every Start has run.
        if (agent == null || !agent.Ready || goals == null || target == null ||
            goals.homeTargets == null || goals.homeTargets.Length != agent.JointCount ||
            goals.jointRanges == null || goals.jointRanges.Length != agent.JointCount)
        {
            Debug.LogError("IkGoalProvider needs a ready ArmAgent, an ArmJointController and a target Transform.", this);
            enabled = false;
            return;
        }

        // The target is only a marker: a collider on it would do nothing for MuJoCo but is easy
        // to confuse with a physical obstacle, so switch any off.
        foreach (var collider in target.GetComponentsInChildren<Collider>())
        {
            Debug.LogWarning($"IkGoalProvider: disabling collider on target '{collider.name}'.", this);
            collider.enabled = false;
        }

        int n = agent.JointCount;
        q = new float[n];
        qRest = new float[n];
        lo = new float[n];
        hi = new float[n];
        savedQ = new double[n];
        dq = new float[n];
        jacobian = new Vector3[n];
        for (int i = 0; i < n; i++)
        {
            qRest[i] = goals.homeTargets[i];
            lo[i] = goals.jointRanges[i].x;
            hi[i] = goals.jointRanges[i].y;
            if (stayInTrainingRegion)
            {
                float span = goals.SampleSpan(i);
                lo[i] = Mathf.Max(lo[i], qRest[i] - span);
                hi[i] = Mathf.Min(hi[i], qRest[i] + span);
            }
        }

        ready = true;
        TakeOverGoals();
    }

    void OnEnable()
    {
        if (ready)
            TakeOverGoals();
    }

    void OnDisable()
    {
        if (tookOverGoals && goals != null)
            goals.randomGoals = previousRandomGoals;
        tookOverGoals = false;
        seeded = false;
    }

    void TakeOverGoals()
    {
        if (tookOverGoals)
            return;
        previousRandomGoals = goals.randomGoals;
        goals.randomGoals = false;
        tookOverGoals = true;
    }

    unsafe void FixedUpdate()
    {
        if (!ready || target == null || !MjScene.InstanceExists)
            return;
        var scene = MjScene.Instance;
        if (scene.Data == null)
            return;

        if (!seeded)
        {
            for (int i = 0; i < q.Length; i++)
                q[i] = Mathf.Clamp((float)scene.Data->qpos[agent.joints[i].QposAddress], lo[i], hi[i]);
            seeded = true;
        }

        Solve(scene, target.position);
        goals.SetTarget(q);
    }

    unsafe void Solve(MjScene scene, Vector3 goal)
    {
        var data = scene.Data;
        int n = q.Length;
        for (int i = 0; i < n; i++)
            savedQ[i] = data->qpos[agent.joints[i].QposAddress];

        float lambda2 = damping * damping;
        Vector3 p = Forward(scene, q);
        for (int iter = 0; iter < iterations; iter++)
        {
            Vector3 e = goal - p;
            if (e.magnitude < tolerance)
                break;

            // Numerical Jacobian: column j is how far the end effector moves per radian of joint j.
            for (int j = 0; j < n; j++)
            {
                float keep = q[j];
                q[j] = keep + finiteDifferenceStep;
                jacobian[j] = (Forward(scene, q) - p) / finiteDifferenceStep;
                q[j] = keep;
            }

            // A = J J^T + lambda^2 I (3x3), solved through Matrix4x4's inverse.
            Matrix4x4 a = Matrix4x4.zero;
            for (int j = 0; j < n; j++)
            {
                Vector3 c = jacobian[j];
                a.m00 += c.x * c.x; a.m01 += c.x * c.y; a.m02 += c.x * c.z;
                a.m11 += c.y * c.y; a.m12 += c.y * c.z; a.m22 += c.z * c.z;
            }
            a.m10 = a.m01; a.m20 = a.m02; a.m21 = a.m12;
            a.m00 += lambda2; a.m11 += lambda2; a.m22 += lambda2;
            a.m33 = 1f;
            Matrix4x4 aInv = a.inverse;

            // Task step: dq = J^T A^-1 e
            Vector3 y = aInv.MultiplyVector(e);
            // Null-space step: z - J^T A^-1 (J z), with z pulling toward the rest posture.
            Vector3 jz = Vector3.zero;
            for (int j = 0; j < n; j++)
                jz += jacobian[j] * (nullSpaceGain * (qRest[j] - q[j]));
            Vector3 yz = aInv.MultiplyVector(jz);

            float largest = 0f;
            for (int j = 0; j < n; j++)
            {
                float z = nullSpaceGain * (qRest[j] - q[j]);
                dq[j] = Vector3.Dot(jacobian[j], y) + z - Vector3.Dot(jacobian[j], yz);
                largest = Mathf.Max(largest, Mathf.Abs(dq[j]));
            }
            float scale = largest > maxStep ? maxStep / largest : 1f;
            for (int j = 0; j < n; j++)
                q[j] = Mathf.Clamp(q[j] + scale * dq[j], lo[j], hi[j]);

            p = Forward(scene, q);
        }

        LastError = (goal - p).magnitude;
        lastEndEffector = p;

        // Restore the real arm's pose so the physics step and other scripts see the true state.
        for (int i = 0; i < n; i++)
            data->qpos[agent.joints[i].QposAddress] = savedQ[i];
        MujocoLib.mj_kinematics(scene.Model, data);
    }

    // Forward kinematics for joint vector `joints` using MuJoCo itself.
    unsafe Vector3 Forward(MjScene scene, float[] joints)
    {
        var data = scene.Data;
        for (int i = 0; i < joints.Length; i++)
            data->qpos[agent.joints[i].QposAddress] = joints[i];
        MujocoLib.mj_kinematics(scene.Model, data);
        return agent.EndEffectorPosition(data);
    }

    void OnDrawGizmos()
    {
        if (!drawGizmos || !ready || target == null)
            return;
        Gizmos.color = LastError < 0.01f ? Color.green : LastError < 0.05f ? Color.yellow : Color.red;
        Gizmos.DrawLine(lastEndEffector, target.position);
        Gizmos.DrawWireSphere(lastEndEffector, 0.015f);
    }
}
