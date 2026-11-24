# CA3 Report

**Name: Aarya Rajgor**  
**Unity version:** 6000.3.24f1  
**MuJoCo plugin version:** 3.17.0  
**Chosen arm (Menagerie):** Universal Robots UR5e (`universal_robots_ur5e/ur5e.xml`)  
**Gallery / repo link:** https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e  

## Why this arm
The UR5e has 6 revolute joints, each driven by a position servo (`general` actuator, kp = 2000 / 500), plus a `home` keyframe and a tool site (`attachment_site`), so the action is simply one joint-angle target per joint. With 6 joints the state is 12-dimensional, smaller than the 7-DOF arms', which keeps the world model and the 78-parameter controller small.

## Pipeline summary
`RolloutRecorder` logged 70,000 Unity/MuJoCo steps at a 0.01 s fixed timestep while `ArmJointController` sent random joint goals (home ± 35% of each half-range, a new goal every 1.5 s) straight to the servos. `train.py` fits an MLP world model that predicts the change in state from `[q, qd, q*]` and exports it to ONNX for Unity's Inference Engine. `train_controller.py` then uses CMA-ES to optimize a linear policy entirely inside that world model. In Unity, `WorldModelController` runs the policy, `WorldModelDriver` shows the model's ghost arm and forecast, and `IkGoalProvider` turns a draggable Cartesian target into joint goals for the controller.

## World model results
- DOF (`ACTION`): 6 (STATE = 12)
- One-step joint RMSE: 0.00896 rad
- Open-loop 10 / 50 / 100-step RMSE: 0.0577 / 0.1088 / 0.1098 rad
- Model: 3 × 256 SiLU MLP. Input is normalized `[q, qd, q*]` plus 6 servo-error inputs `(q* − q) / 0.1` computed inside the network; output is normalized `s' − s`. Loss is one-step MSE + 0.5 × a multi-step open-loop loss whose horizon grows from 2 to 16 steps. Trained for 40 epochs on the first 90% of the rollout and evaluated on the last 10%.
- Data check: replaying each recorded (state, action) pair in Python MuJoCo reproduced Unity's next state to within 3.6e-7 rad, which confirms the action recorded at step t is the one that produced step t+1.

## Controller results
- Imagined return (CMA-ES): -3.257 (held-out; open-loop baseline also -3.257, so θ = 0 was kept. Best sample -3.291, CMA mean -3.272)
- Policy: `a = goal + W·[q − goal, qd] + b` (W is 6 × 12, b is 6), followed by the same low-pass (0.2) and 0.05 rad/step rate limit used in imagination. θ = 0 is plain goal commanding.
- Qualitative joint-goal tracking: with the random training-style goals, the arm moves smoothly to each new pose and settles before the next goal arrives. The rate limit makes it noticeably slower and smoother than the raw step commands used during data collection. In an offline MuJoCo check of the same controller it took about 1.3 s to get within 0.01 rad of a goal (1.06 s for raw step commands), and the remaining error at rest was about 0.006 rad, which is gravity sag in the position servos.
- Why CMA-ES kept θ = 0: with the UR5e's stiff servos and the action smoothing, the policy has little room to change the cost. Hand-set gains scored the same way in imagination and in real MuJoCo (within about 1%), and none beat θ = 0. A variant with actions relative to the current pose did learn to reach from scratch (imagined return −17.2 → −3.4), but it settled about 0.04 rad off target in reality, so it was not used.

## WorldModelDriver
- Predict / reanchor / forecast notes:
  - **Predict** normalizes `[s, a]` with `norm.json`, runs `dynamics.onnx` on Inference Engine's CPU backend, and returns `s + Δ·y_std + y_mean`. In a headless Unity test, Inference Engine matched PyTorch's output for this architecture to within 2.4e-6. In Play mode one call costs about 0.09 ms, so the ghost and forecast run at full speed.
  - **The ghost** is stepped once per physics tick with the action the real arm actually executed, so it shows the model's prediction for identical inputs. Each step it is blended 5% toward the real state.
  - **Reanchor:** it snaps back onto the real arm every 3000 steps, or as soon as the joint RMS error exceeds 0.2 rad.
  - **Forecast:** every 10 steps it imagines 100 steps ahead using the learned policy with the same smoothing, and draws the predicted tool path. A deviation line connects the ghost's tool to the real tool.

## IK + live target
- IK method: damped least squares (Levenberg–Marquardt) on the `attachment_site` position, λ = 0.05.
  - **Jacobian:** 3 × 6, built from finite differences of MuJoCo's own `mj_kinematics`, so it always matches the imported model and avoids Unity/MuJoCo frame conversions.
  - **Extra freedom:** a position target fixes only 3 of the 6 joints, so a null-space term pulls the spare freedom toward the home posture.
  - **Joint limits:** solutions are clamped to home ± the random-goal span, the region the world model was trained on.
  - **Warm start:** the solver starts from the previous tick's solution. Its joint answer becomes the ArmJointController goal that the learned controller tracks.
- Workspace / failure cases:
  - **Normal targets:** reachable targets in front of the arm converge to millimeter accuracy. In an offline test of the same algorithm, 97% of random reachable targets solved to under 5 mm from a cold start (median 0.7 mm), and a target moving 1 cm per step was tracked within 2 mm.
  - **Out of reach (beyond about 0.85 m):** the arm stretches toward the target without becoming unstable, and the error gizmo turns red.
  - **Outside the trained joint region** (for example behind or underneath the base): the solution clamps at the region edge and leaves a residual error. This is deliberate, because those poses would put the world model and controller outside their training data.

## Design question
Why can rollout collection stay in joint space while deployment tracks a Cartesian target?

The world model and controller only describe dynamics: how joint angles and velocities respond to joint targets. That relationship doesn't depend on how a goal was chosen, so joint-space rollouts are enough to learn it. Turning a Cartesian target into joint goals is a kinematics problem. The arm's geometry is known exactly, and IK solves it with MuJoCo's forward kinematics, with no learning and no extra data. Deployment therefore just puts IK in front of the same joint-goal interface the controller was trained on. The one condition is that IK's joint solutions stay inside the region the rollouts covered, which is why `IkGoalProvider` clamps to that region.

## AI assistance
I used Claude (Anthropic), an AI coding assistant, through Claude Code while completing this assignment. It:

- set up the Unity 6 project with the MuJoCo 3.17.0 plugin and Unity Inference Engine, and chose the 0.01 s fixed timestep after finding that the UR5e's servos oscillate at 0.02 s;
- wrote the implementations of an editor helper (CA3SceneSetup.cs) that wires up the scene for each step;
- tested the pipeline offline in Python MuJoCo, chose the world-model architecture and settings, and found and fixed the skeleton's malformed .meta files;
- guided me through the Unity steps and debugged problems along the way (the Pause toggle, the importer's ur5e343 naming, scripts reloading during Play);
- wrote this report from the measured results;
- recorded the demo video and organized the files for submission.
