import json
from pathlib import Path
import time
import cma
import numpy as np
import torch
from torch import nn

from train import build_model

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 12
ACTION = 6
FEAT = STATE
N_PARAMS = ACTION * FEAT + ACTION

# UR5e values from ur5e.xml: <key name="home"> and the joint ranges.
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float32)
JOINT_LO = np.array([-6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319], dtype=np.float32)
JOINT_HI = np.array([6.28319, 6.28319, 3.1415, 6.28319, 6.28319, 6.28319], dtype=np.float32)
RANDOM_SCALE = 0.35   # must match ArmJointController.randomScale

VEL_PENALTY = 0.05
ACTION_SMOOTH_PENALTY = 0.1
ACTION_SMOOTH = 0.2   # must match WorldModelController.actionSmoothing
MAX_DELTA = 0.05      # must match WorldModelController.maxDeltaPerStep
HORIZON = 200         # 2 s at the 0.01 s fixed timestep
EPISODES = 16
CMA_ITERS = 60
SIGMA0 = 0.05        # search scale; theta = 0 is already a good controller, so stay near it
EVAL_SEEDS = range(900_000, 900_005)   # held-out goals used to pick the controller we save


def build_dynamics():
    model = build_model()  # same MLP as train.py
    path = ROOT / "dynamics.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_norm():
    path = ROOT / "norm.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    data = np.load(path)
    return data["x_mean"], data["x_std"], data["y_mean"], data["y_std"]


def unpack(theta):
    w = theta[: ACTION * FEAT].reshape(ACTION, FEAT)
    b = theta[ACTION * FEAT :]
    return w, b


def policy(state, goal, w, b):
    """Linear residual policy on the goal:
        features = [q - goal, qd]
        action   = goal + W @ features + b
    theta = 0 is plain open-loop goal commanding (what the data was collected with); CMA-ES learns
    how much to lead or damp the position servos so the arm settles on the goal faster."""
    q = state[..., :ACTION]
    qd = state[..., ACTION:]
    features = np.concatenate([q - goal, qd], axis=-1)
    raw = goal + features @ w.T + b
    return np.clip(raw, JOINT_LO, JOINT_HI).astype(np.float32)


@torch.no_grad()
def step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std):
    x = np.concatenate([state, action], axis=-1).astype(np.float32)
    x_n = (x - x_mean) / x_std
    delta = model(torch.as_tensor(x_n[None] if x_n.ndim == 1 else x_n)).numpy()
    delta = delta.reshape(np.shape(state)) * y_std + y_mean
    return (state + delta).astype(np.float32)


def sample_goal(rng, n=None):
    span = (JOINT_HI - JOINT_LO) * 0.5 * RANDOM_SCALE
    size = None if n is None else (n, ACTION)
    goal = HOME + rng.uniform(-span, span, size=size)
    return np.clip(goal, JOINT_LO, JOINT_HI).astype(np.float32)


def smooth_action(prev, raw):
    blended = (1.0 - ACTION_SMOOTH) * prev + ACTION_SMOOTH * raw
    delta = np.clip(blended - prev, -MAX_DELTA, MAX_DELTA)
    return np.clip(prev + delta, JOINT_LO, JOINT_HI).astype(np.float32)


def rollout(model, theta, x_mean, x_std, y_mean, y_std, rng, episodes=1):
    """Imagined episodes inside the world model, run side by side as one batch. Each arm starts
    at rest at a random training-style pose and has HORIZON steps to reach another random goal.
    Returns the mean reward per step of each episode (higher is better)."""
    w, b = unpack(theta)
    start = sample_goal(rng, episodes)
    goal = sample_goal(rng, episodes)
    state = np.concatenate([start, np.zeros_like(start)], axis=-1)
    prev = start.copy()   # WorldModelController seeds its smoothing with the current pose
    total = np.zeros(episodes)
    for _ in range(HORIZON):
        action = smooth_action(prev, policy(state, goal, w, b))
        state = step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std)
        if not np.all(np.isfinite(state)) or np.abs(state).max() > 1e3:
            return np.full(episodes, -1e6)  # the policy drove the model somewhere it cannot predict
        q, qd = state[:, :ACTION], state[:, ACTION:]
        cost = (np.sum((q - goal) ** 2, axis=-1)
                + VEL_PENALTY * np.sum(qd ** 2, axis=-1)
                + ACTION_SMOOTH_PENALTY * np.sum((action - prev) ** 2, axis=-1))
        total -= cost
        prev = action
    return total / HORIZON


def fitness(theta, model, x_mean, x_std, y_mean, y_std, seed):
    rng = np.random.default_rng(seed)
    returns = rollout(model, theta, x_mean, x_std, y_mean, y_std, rng, episodes=EPISODES)

    return -float(np.mean(returns))


def held_out_return(theta, model, norms):
    return -float(np.mean([fitness(theta, model, *norms, seed) for seed in EVAL_SEEDS]))


def save_controller(theta, imagined_return=None, baseline_return=None):
    w, b = unpack(theta)
    payload = {
        "feature": "q_minus_goal_and_qd",
        "state_size": STATE,
        "action_size": ACTION,
        "W": w.astype(float).reshape(-1).tolist(),
        "b": b.astype(float).tolist(),
        "joint_lo": JOINT_LO.astype(float).tolist(),
        "joint_hi": JOINT_HI.astype(float).tolist(),
        "home": HOME.astype(float).tolist(),
        "vel_penalty": VEL_PENALTY,
        "horizon": HORIZON,
        "action_smooth": ACTION_SMOOTH,
        "max_delta": MAX_DELTA,
        "imagined_return": imagined_return,
        "baseline_return": baseline_return,
    }
    path = ROOT / "controller.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def main():
    torch.set_num_threads(1)  # batch-of-one inference is faster without thread overhead
    model = build_dynamics()
    norms = load_norm()

    theta0 = np.zeros(N_PARAMS)
    baseline = held_out_return(theta0, model, norms)
    print(f"open-loop baseline (theta = 0) imagined return: {baseline:.4f}")

    es = cma.CMAEvolutionStrategy(theta0, SIGMA0, {"seed": 1, "verbose": -9})
    for it in range(CMA_ITERS):
        solutions = es.ask()
        # Common random numbers: every candidate in a generation faces the same goals.
        seed = 1000 + it
        losses = [fitness(np.asarray(s), model, *norms, seed) for s in solutions]
        es.tell(solutions, losses)
        print(f"iter {it + 1:3d}/{CMA_ITERS}  best {-min(losses):.4f}  "
              f"mean {-float(np.mean(losses)):.4f}  sigma {es.sigma:.4f}")

    candidates = {
        "baseline": theta0,
        "best sample": np.asarray(es.result.xbest),
        "distribution mean": np.asarray(es.result.xfavorite),
    }
    scores = {name: held_out_return(theta, model, norms) for name, theta in candidates.items()}
    for name, score in scores.items():
        print(f"held-out imagined return, {name}: {score:.4f}")
    pick = max(scores, key=scores.get)
    print(f"saving {pick}")
    if pick == "baseline":
        print("note: no CMA-ES candidate beat plain goal commanding on held-out imagined goals, so the\n"
              "      saved policy is theta = 0 (command the goal, then smooth). With stiff position servos\n"
              "      the smoothing and rate limit leave the policy little room to help; see CA3_SETUP.md.")
    save_controller(candidates[pick], imagined_return=scores[pick], baseline_return=baseline)


if __name__ == "__main__":
    current_time = time.time()
    main()
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
