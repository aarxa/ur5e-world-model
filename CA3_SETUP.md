# CA3 setup and run guide (UR5e)

Project: `~/Documents/CS6983_CA3` (Unity 6000.3.24f1, built-in render pipeline).
Arm: Universal Robots UR5e from MuJoCo Menagerie (`MJCF/universal_robots_ur5e/ur5e.xml`).

Already done in the project:

- `Packages/manifest.json` references the MuJoCo plugin **3.17.0** (`External/mujoco-3.17.0/unity`, checked out at the `3.17.0` tag) and Unity Inference Engine **2.6.1** (needed by `WorldModelDriver` and `CA3.asmdef`).
- **Fixed Timestep is 0.01 s** (Project Settings → Time). At Unity's default 0.02 s, the UR5e's stiff position servos (kp = 2000) sometimes oscillate and never settle on a goal. In a 300-second test, 3 of 100 goals were still 0.4–1.1 rad off after 3 s. At 0.01 s every goal settles to under 0.011 rad.
- The CA3 scripts are in `Assets/CA3`. Python training uses the venv in `.venv` (torch, cma, onnx).
- **Skeleton bug fixed:** in `CA3_Scripts-1.zip`, the `.meta` files for `WorldModelController.cs`, `WorldModelDriver.cs` and `IkGoalProvider.cs` have malformed GUIDs (26–28 characters instead of 32). Unity silently ignores those three scripts, and they never compile. The bad `.meta` files were deleted and Unity regenerated valid ones. If you re-copy the skeleton, delete those three `.meta` files again.

## 1. Install the MuJoCo 3.17.0 native library (one time)

1. Download `mujoco-3.17.0-macos-universal2.dmg` from https://github.com/google-deepmind/mujoco/releases/tag/3.17.0
2. Drag **MuJoCo.app** into `/Applications`, then open it once (right-click → Open) so macOS trusts the library.
3. Copy the library into the plugin package:

```bash
cp /Applications/MuJoCo.app/Contents/Frameworks/mujoco.framework/Versions/Current/libmujoco.3.17.0.dylib ~/Documents/CS6983_CA3/External/mujoco-3.17.0/unity/mujoco.dylib
```

4. In Unity Hub: **Add → Add project from disk →** `~/Documents/CS6983_CA3`, then open it.

## Shortcut: the CA3 menu

`Assets/Editor/CA3SceneSetup.cs` adds a **CA3** menu to Unity that does the Inspector wiring in steps 2–6 for you:

| Menu item | What it does |
|---|---|
| Make Ghost Prefab + Materials | Saves the scene as `Assets/CA3/CA3.unity`, saves the imported arm as `ur5e_ghost.prefab` (the scene arm stays a plain object), and creates `Ghost.mat` (Standard, Fade) and `Line.mat` (Sprites/Default). |
| Set Up Rollout Recording | Adds ArmAgent, ArmJointController and RolloutRecorder to the arm and turns on random goals and open-loop actions. |
| Set Up Learned Controller + Ghost (Step 3) | Disables the recorder and adds WorldModelController (`controller.json`) and WorldModelDriver (`dynamics.onnx`, `norm.json`, ghost prefab, materials). |
| Set Up IK Tracking (Step 4) | Creates a collider-free `TrackingTarget` sphere at the tool's home position and adds IkGoalProvider aimed at it. |
| Record IK Demo Video | Enters Play mode, moves `TrackingTarget` around a loop in front of the arm, and records the Game view to `Recordings/ur5e_ik_demo.mp4` with Unity Recorder. |

The menu finds the arm through `shoulder_pan_joint`, because the importer names the root after the file plus a number (`ur5e343` here).

**Two Unity gotchas hit during setup:**

- **Pause stays on.** If the ‖ button is lit when you press Play, Unity runs one frame and freezes, which looks like the arm not moving. The CA3 editor script switches Pause off when Play starts and logs that it did. It also turns on Run In Background, so Play keeps going while another app is in front.
- **Don't let scripts reload during Play.** If Unity recompiles while playing, the MuJoCo plugin loses its scene and floods the Console with "A MjScene singleton is created automatically, yet multiple instances exist". Press Stop before editing scripts, or set **Unity → Settings → General → Script Changes While Playing** to *Recompile After Finished Playing*.

## 2. Import the arm

1. **Assets → Import MuJoCo Scene** → pick `MJCF/universal_robots_ur5e/ur5e.xml`.
   This creates a root GameObject named `ur5e` plus a number, such as `ur5e343` (bodies, hinge joints, 6 actuators, `attachment_site`).
2. Check the wiring: each `MjActuator` under `ur5e` has its **Joint** set (`shoulder_pan` → `shoulder_pan_joint`, …), and **MjGlobalSettings** shows integrator `implicitfast`.
3. **Make the ghost prefab now, before adding scripts:** drag `ur5e` from the Hierarchy into `Assets/CA3` to create a prefab, rename the prefab asset to `ur5e_ghost`, and keep the scene object (it is the real arm). If Unity turns the scene object into a prefab instance, that is fine.
4. Create two materials in `Assets/CA3`:
   - `Ghost`: Standard shader, Rendering Mode **Fade**, light blue, alpha about 0.3.
   - `Line`: shader **Sprites/Default**, any bright color.

Values copied from `ur5e.xml` into the scripts:

| | Value |
|---|---|
| Joints (6) | `shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint` |
| Actuators (6, position servos) | `shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3` |
| End effector | body `wrist_3_link`, tool site `attachment_site` |
| Joint ranges (rad) | ±6.28319 for all, except the elbow at ±3.1415 |
| Home (`<key name="home">`) | `-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0` |
| STATE / ACTION | 12 (6 angles + 6 velocities) / 6 |

## 3. Collect rollouts (Step 2)

1. Select `ur5e` and add **ArmAgent**, **ArmJointController** and **RolloutRecorder**. They find the joints, actuators and end effector by MJCF name, so no drag-and-drop wiring is needed.
2. Set ArmJointController **Emit Actions = on** and **Random Goals = on** (the defaults).
3. Press Play. The arm moves to home, then to a new random goal every 1.5 s. The recorder runs at 4× speed. After 70,000 steps (700 s of simulated time, about 3 minutes of wall time) it writes `Assets/CA3/Data/rollout.csv` and stops Play mode.
   Holding each goal 1.5 s instead of 3 s gives the world model more data where the arm is moving. In testing, that cut its 50-step prediction error by about 25%.
   Columns: `q0..5, qd0..5, qstar0..5, qnext0..5, qdnext0..5`. The `#` header line is skipped by `np.loadtxt`.

## 4. Train the world model and controller

```bash
cd ~/Documents/CS6983_CA3/Assets/CA3 && ../../.venv/bin/python train.py
```

```bash
cd ~/Documents/CS6983_CA3/Assets/CA3 && ../../.venv/bin/python train_controller.py
```

`train.py` writes `Data/dynamics.pt`, `norm.npz`, `norm.json` and `dynamics.onnx`, and prints the one-step and 10/50/100-step open-loop RMSE for `REPORT.md`.
`train_controller.py` writes `Data/controller.json` and prints the imagined return of the learned policy next to the open-loop baseline.

## 5. Check the controller on the training-style goals (Step 3.3)

On `ur5e`:

1. Disable **RolloutRecorder**.
2. Add **WorldModelController** and set **Controller Json** to `Data/controller.json`. Keep Action Smoothing at 0.2 and Max Delta at 0.05; they must match `train_controller.py`.
3. Add **WorldModelDriver**: **Model Asset** = `Data/dynamics.onnx`, **Norm Json** = `Data/norm.json`, **Ghost Arm** = the `ur5e_ghost` prefab, **Ghost Material** = `Ghost`, **Line Material** = `Line`.
4. Leave ArmJointController **Random Goals = on**. WorldModelController switches Emit Actions off by itself.
5. Press Play. Random goals keep coming, and the learned policy drives the arm toward each one. You should see:
   - the ghost arm, which is the world model's prediction given the same commands;
   - the forecast line, a 100-step imagined end-effector path;
   - the deviation line from the ghost's end effector to the real one.

## 6. IK to a movable target (Step 4)

1. **GameObject → 3D Object → Sphere**, rename it `TrackingTarget`, **remove its Sphere Collider**, and set the scale to 0.05.
   A good starting position is about **(-0.13, 0.49, 0.49)**: the tool flange at the home pose, if `ur5e` sits at the origin.
2. Add **IkGoalProvider** to `ur5e` and set **Target** to `TrackingTarget`. It turns Random Goals off while enabled.
3. Press Play and drag `TrackingTarget` around in the Scene view. IK writes joint goals into ArmJointController, and WorldModelController tracks them. The gizmo line from the flange to the target is green under 1 cm, yellow under 5 cm and red beyond that.
   With **Stay In Training Region** on (the default), IK solutions stay inside home ± the random-goal span. That is the part of joint space the world model has seen, so targets that need poses outside it are reached only as closely as that region allows.

## Script execution order (one physics tick)

`IkGoalProvider (-20)` → `ArmJointController (-10)` → `WorldModelController (-5)` → `ArmAgent (-2, writes ctrl)` → `RolloutRecorder (-1, logs s, a)` → `MjScene (0, mj_step)` → `WorldModelDriver (10, steps the ghost with the same action)`.

`MjActuator.Control` is only copied into MuJoCo after `mj_step`. `ArmAgent` therefore also writes `data->ctrl` directly, so each recorded action is the one that produced the recorded next state.

## How the pieces work (reference for REPORT.md)

These notes describe what the code does. They are not answers to the reflection questions: those ask about your own experience, so write them from your own run.

**Data (`RolloutRecorder`).** Every 0.01 s physics step logs `(q, qd)`, the commanded joint targets `q*`, and the resulting `(q', qd')`. Goals are `home ± (range/2) × 0.35` per joint, held for 1.5 s each, and sent straight to the position servos (open loop). There is one unbroken run, so row *t*'s next state is row *t+1*'s state, and `train.py` checks this.

**World model (`train.py`).** An MLP maps normalized `[q, qd, q*]` (18 numbers) to the normalized change in state `s' − s` (12 numbers). Inside the network the servo error `q* − q` is rebuilt and added as 6 extra inputs. Those feed 3 hidden layers of 256 SiLU units and a linear output. Loss = one-step MSE + 0.5 × multi-step loss. The multi-step loss (`multi_step_loss`) feeds the model its own predictions for up to 16 steps using the recorded actions, and penalizes drift at every step, in units of each state dimension's standard deviation. The horizon grows from 2 to 16 over the first half of training. The checkpoint with the best validation 50-step error is exported to `dynamics.onnx`. Unity normalizes with `norm.json`, runs the network, then de-normalizes and adds the delta (`WorldModelDriver.Predict`).

**Controller (`train_controller.py`, `WorldModelController`).** A linear policy on the goal: `a = goal + W·[q − goal, qd] + b`, with `W` 6×12 and `b` 6, so 78 parameters. With θ = 0 this is plain goal commanding, the same thing the data was collected with. Every step the action is low-passed (weight 0.2 on the new value) and rate-limited to 0.05 rad. CMA-ES searches θ entirely inside the world model. Each candidate runs 16 imagined episodes of 200 steps, from a random rest pose to a random goal. Reward per step: −(‖q − goal‖² + 0.05‖qd‖² + 0.1‖Δa‖²). Every candidate in a generation sees the same goals. After 60 generations the script compares the baseline, the best sample and the CMA mean on held-out imagined goals, and saves whichever scores best. Unity runs the same policy and the same smoothing every FixedUpdate.

**IK (`IkGoalProvider`).** Damped least squares, also called Levenberg–Marquardt, on the `attachment_site` position. The 3×6 Jacobian comes from finite differences of MuJoCo's own forward kinematics (`mj_kinematics`). That means the Jacobian always matches the imported model and avoids the Unity/MuJoCo handedness flip. The arm has 6 joints but a position target fixes only 3 numbers, so a null-space term pulls the extra freedom toward the home posture. Solutions are clamped to the region the rollouts covered. The solver warm-starts from the previous step's answer. It writes its joint solution to `ArmJointController.SetTarget`, and the learned controller tracks that goal.

**Why joint-space data is enough for a Cartesian target.** The world model and controller only ever see joint angles, velocities and joint targets. IK is a separate, model-free kinematic map from a Cartesian point to a joint goal, so a new kind of goal never requires new dynamics data. It just has to produce joint goals inside the range the data covered, which is what Stay In Training Region enforces.

## What was tested before handoff (and what was not)

Tested outside Unity, with a Python MuJoCo stand-in for the Unity recorder: same `ur5e.xml`, 0.01 s step, the same random-goal process. These numbers come from simulated data, not from your Unity run, so don't copy them into the report.

| Check | Result |
|---|---|
| World model, held-out 1-step / 10 / 50 / 100-step joint RMSE | 0.0083 / 0.052 / 0.087 / 0.086 rad |
| Same metrics, original Tanh MLP with 3 s goals | 0.0139 / 0.079 / 0.152 / 0.157 rad |
| `dynamics.onnx` vs PyTorch (onnx reference runtime) | max difference 4e-7 |
| `dynamics.onnx` in Unity Inference Engine 2.6.1 (headless) | 12 outputs, max difference 2.4e-6 vs PyTorch |
| IK port, 300 random reachable targets, cold start | median error 0.7 mm, 97% under 5 mm, about 8 iterations |
| IK, target moving 1 cm per step, warm start | worst error 2 mm, under 1 iteration per step |
| Unity compile of all 6 CA3 scripts with plugin 3.17.0 + Inference Engine 2.6.1 | 0 errors, 0 warnings |
| `train.py` + `train_controller.py` on a 70k-step rollout | about 66 s + 16 s on this Mac |

**Controller finding.** With the UR5e's stiff position servos and the skeleton's smoothing (0.2), rate limit (0.05 rad/step) and velocity penalty, plain goal commanding (θ = 0) is already near-optimal. Hand-set gains scored in imagination and in real MuJoCo agreed within about 1% and ranked the same way, and none beat θ = 0 by a measurable margin. CMA-ES also found nothing better on held-out goals, so `train_controller.py` saved θ = 0 and printed a note. Two alternatives were tried and rejected:

- An action relative to the current pose (`a = q + W·f + b`) makes CMA-ES learn reaching from scratch: imagined return goes from −17.2 to −3.4. But the learned policy settled 0.04 rad off target in real MuJoCo, against 0.005 rad for goal commanding.
- Training on small moves only did not change the result either.

Your Unity data may differ, and if CMA-ES does find an improvement there, the script will keep it. Not tested: anything inside a live Unity scene (import, Play-mode wiring, the ghost visuals, the recorder's CSV). That needs the MuJoCo 3.17.0 native library, which is installed in step 1.
