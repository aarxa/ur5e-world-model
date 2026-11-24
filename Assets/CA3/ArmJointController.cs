using UnityEngine;

// Owns the current joint-space goal.
//   randomGoals = true  -> resample a goal around homeTargets every holdSeconds (data collection,
//                          and Step 3's "replay the training targets" check).
//   emitActions = true  -> send the goal straight to the arm as the open-loop action.
// The learned controller turns emitActions off and reads CurrentTarget instead; the IK provider
// turns randomGoals off and writes the goal through SetTarget.
[DefaultExecutionOrder(-10)]
public class ArmJointController : MonoBehaviour
{
    public ArmAgent agent;
    [Tooltip("Seconds per random goal. 1.5 s (most goals settle in about 1 s) gives the world model more " +
             "moving data than 3 s, which is mostly the arm sitting still: 50-step open-loop error " +
             "dropped from 0.112 to 0.084 rad in testing.")]
    public float holdSeconds = 1.5f;

    // UR5e joint limits from ur5e.xml (radians). The elbow uses the size3_limited class.
    public Vector2[] jointRanges =
    {
        new Vector2(-6.28319f, 6.28319f),
        new Vector2(-6.28319f, 6.28319f),
        new Vector2(-3.1415f, 3.1415f),
        new Vector2(-6.28319f, 6.28319f),
        new Vector2(-6.28319f, 6.28319f),
        new Vector2(-6.28319f, 6.28319f),
    };

    // <key name="home"> from ur5e.xml.
    public float[] homeTargets = { -1.5708f, -1.5708f, 1.5708f, -1.5708f, -1.5708f, 0f };

    [Range(0f, 1f)]
    public float randomScale = 0.35f;

    public bool emitActions = true;

    [Tooltip("Resample random goals every holdSeconds. The IK provider switches this off while it drives the goal.")]
    public bool randomGoals = true;

    [Tooltip("Seed for the goal sampler; -1 picks a new seed every run.")]
    public int randomSeed = 0;

    float[] target;
    float hold;
    bool ready;
    System.Random rng;

    public float[] CurrentTarget => target;
    public bool Ready => ready;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (agent == null || !agent.Ready)
        {
            Debug.LogError("ArmJointController needs a ready ArmAgent on the same arm.", this);
            enabled = false;
            return;
        }
        if (homeTargets == null || homeTargets.Length != agent.ActionSize ||
            jointRanges == null || jointRanges.Length != agent.ActionSize)
        {
            Debug.LogError(
                $"ArmJointController: homeTargets and jointRanges need {agent.ActionSize} entries " +
                "(one per actuator).", this);
            enabled = false;
            return;
        }

        rng = randomSeed >= 0 ? new System.Random(randomSeed) : new System.Random();
        target = new float[agent.ActionSize];
        PlaceTarget(true);
        hold = holdSeconds;
        ready = true;
    }

    void FixedUpdate()
    {
        if (!ready)
            return;

        if (randomGoals)
        {
            hold -= Time.fixedDeltaTime;
            if (hold <= 0f)
            {
                PlaceTarget(false);
                hold = holdSeconds;
            }
        }

        if (emitActions)
            agent.SetAction(target);
    }

    // Open-loop action: command the goal itself.
    public void ActionFrom(float[] destination)
    {
        if (target == null || destination == null)
            return;
        int count = Mathf.Min(destination.Length, target.Length);
        for (int i = 0; i < count; i++)
            destination[i] = target[i];
    }

    public void PlaceTarget(bool home)
    {
        if (target == null)
            return;
        for (int i = 0; i < target.Length; i++)
        {
            float goal = homeTargets[i];
            if (!home)
            {
                float span = SampleSpan(i);
                goal += (float)(rng.NextDouble() * 2.0 - 1.0) * span;
            }
            target[i] = Mathf.Clamp(goal, jointRanges[i].x, jointRanges[i].y);
        }
    }

    // Goal from an external source (IK). Clamped to the joint limits.
    public void SetTarget(float[] goal)
    {
        if (target == null || goal == null)
            return;
        int count = Mathf.Min(goal.Length, target.Length);
        for (int i = 0; i < count; i++)
            target[i] = Mathf.Clamp(goal[i], jointRanges[i].x, jointRanges[i].y);
    }

    // Half-width of the region random goals are drawn from, matching sample_goal() in
    // train_controller.py: (hi - lo) / 2 * randomScale.
    public float SampleSpan(int joint)
    {
        return (jointRanges[joint].y - jointRanges[joint].x) * 0.5f * randomScale;
    }
}
