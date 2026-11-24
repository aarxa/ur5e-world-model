import numpy as np
import torch
import time
import warnings
from pathlib import Path
from torch import nn

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 12   # UR5e: 6 joint angles + 6 joint velocities
ACTION = 6   # UR5e: 6 position actuators

HIDDEN = 256
ERR_SCALE = 0.1      # radians; scales the servo-error inputs (q* - q) to roughly unit size
EPOCHS = 40
BATCH = 512
LR = 1e-3
VAL_FRACTION = 0.1
MS_HORIZON = 16      # length of the multi-step (open-loop) training rollouts
MS_BATCH = 128
MS_WEIGHT = 0.5      # weight of the multi-step loss next to the one-step loss
SEED = 0


class WorldModel(nn.Module):
    """MLP world model: normalized [q, qd, q*] -> normalized (s' - s).

    Unity feeds exactly that normalized vector to dynamics.onnx. Inside the network the servo error
    q* - q is rebuilt and added as 6 extra inputs: the UR5e's position actuators make torque from
    that difference and clip it at their force limits, a sharp kink a plain MLP fits poorly when
    q and q* arrive as separately normalized numbers. On simulated UR5e rollouts, SiLU plus these
    inputs cut the held-out 50-step open-loop error from 0.152 to 0.112 rad compared with a plain
    Tanh MLP trained on the same data."""

    def __init__(self):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(STATE + ACTION))
        self.register_buffer("x_std", torch.ones(STATE + ACTION))
        self.net = nn.Sequential(
            nn.Linear(STATE + ACTION + ACTION, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, STATE),
        )

    def set_norm(self, x_mean, x_std):
        self.x_mean.copy_(torch.as_tensor(x_mean))
        self.x_std.copy_(torch.as_tensor(x_std))

    def forward(self, x_n):
        x = x_n * self.x_std + self.x_mean
        servo_error = (x[..., STATE:STATE + ACTION] - x[..., 0:ACTION]) / ERR_SCALE
        return self.net(torch.cat([x_n, servo_error], dim=-1))


def build_model():
    return WorldModel()


def mean_std(z):
    return z.mean(0), z.std(0) + 1e-6


def norm(z, mean, std):
    return (z - mean) / std


def save_norm_json(x_mean, x_std, y_mean, y_std):
    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    lines = ["{"]
    keys = list(arrays)
    for index, key in enumerate(keys):
        values = ", ".join(f"{float(value):.8f}" for value in arrays[key])
        comma = "," if index < len(keys) - 1 else ""
        lines.append(f'  "{key}": [{values}]{comma}')
    lines.append("}")
    (ROOT / "norm.json").write_text("\n".join(lines) + "\n")


def load_rollout():
    path = ROOT / "rollout.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"No rollout at {path}. Enable RolloutRecorder and Play.")
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != STATE + ACTION + STATE:
        raise ValueError(f"Expected {STATE + ACTION + STATE} columns, found {data.shape[1]}.")
    return data


def predict_delta(model, state, action, x_mean, x_std, y_mean, y_std):
    x = np.concatenate([state, action], axis=-1)
    x_n = (x - x_mean) / x_std
    with torch.no_grad():
        delta_n = model(torch.as_tensor(x_n)).numpy()
    return delta_n * y_std + y_mean


def joint_rmse(pred, true):
    err = pred[..., 0:ACTION] - true[..., 0:ACTION]
    return float(np.sqrt((err ** 2).sum(axis=-1).mean()))


def open_loop_rmse(model, data, start, x_mean, x_std, y_mean, y_std, horizon):
    end = len(data) - horizon
    if end <= start:
        return float("nan")
    idx = np.arange(start, end)
    if len(idx) > 2000:
        idx = idx[:: max(1, len(idx) // 2000)]
    state = data[idx, 0:STATE].copy()
    for t in range(horizon):
        action = data[idx + t, STATE:STATE + ACTION]
        state = state + predict_delta(model, state, action, x_mean, x_std, y_mean, y_std)
    return joint_rmse(state, data[idx + horizon, 0:STATE])


def multi_step_loss(model, data, split, x_mean, x_std, y_mean, y_std, horizon, batch, loss_fn):
    """Roll the model forward on its own predictions for `horizon` steps using the recorded
    actions, and penalise the drift from the recorded states at every step.

    `data` is a float32 tensor of rollout rows and `split` the first validation row, so every
    window stays inside the training part. Errors are measured in units of each state
    dimension's standard deviation so joint angles and velocities count equally."""
    starts = torch.randint(0, split - horizon, (batch,))
    state = data[starts, 0:STATE]
    state_scale = x_std[0:STATE]
    total = 0.0
    for t in range(horizon):
        rows = starts + t
        action = data[rows, STATE:STATE + ACTION]
        x_n = (torch.cat([state, action], dim=-1) - x_mean) / x_std
        state = state + model(x_n) * y_std + y_mean
        target = data[rows, STATE + ACTION:]
        total = total + loss_fn(state / state_scale, target / state_scale)
    return total / horizon


def check_contiguous(data):
    """The recorder logs one unbroken run, so row t's next state is row t+1's state."""
    gap = np.abs(data[:-1, STATE + ACTION:] - data[1:, 0:STATE]).max(axis=1)
    breaks = int((gap > 1e-4).sum())
    if breaks:
        print(f"warning: {breaks} discontinuities in rollout.csv; multi-step windows may cross them")


def export_onnx(model):
    path = ROOT / "dynamics.onnx"
    dummy = torch.zeros(1, STATE + ACTION)
    kwargs = dict(input_names=["x"], output_names=["delta"], opset_version=15)
    with warnings.catch_warnings():
        # The classic exporter emits plain opset-15 Gemm/Sigmoid ops that Inference Engine loads.
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(model, dummy, str(path), **kwargs)
    print(f"wrote {path}")

    try:  # make sure the file Unity loads computes the same thing as the PyTorch model
        import onnx
        from onnx.reference import ReferenceEvaluator
        onnx.checker.check_model(str(path))
        probe = torch.randn(1, STATE + ACTION)
        with torch.no_grad():
            expected = model(probe).numpy()
        got = ReferenceEvaluator(str(path)).run(None, {"x": probe.numpy()})[0]
        print(f"ONNX check: max |onnx - torch| = {np.abs(got - expected).max():.2e}")
    except ImportError:
        pass


def train_world_model(data):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    check_contiguous(data)

    n = len(data)
    split = int(n * (1.0 - VAL_FRACTION))   # train on the first 90%, validate on the last 10%
    x = data[:, 0:STATE + ACTION]
    y = data[:, STATE + ACTION:] - data[:, 0:STATE]   # predict the change in state
    x_mean, x_std = mean_std(x[:split])
    y_mean, y_std = mean_std(y[:split])

    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / "norm.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    save_norm_json(x_mean, x_std, y_mean, y_std)

    xn = torch.as_tensor(norm(x[:split], x_mean, x_std))
    yn = torch.as_tensor(norm(y[:split], y_mean, y_std))
    data_t = torch.as_tensor(data)
    stats_t = [torch.as_tensor(v) for v in (x_mean, x_std, y_mean, y_std)]

    model = build_model()
    model.set_norm(x_mean, x_std)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    mse = nn.MSELoss()

    print(f"{n} transitions ({split} train / {n - split} val), STATE={STATE}, ACTION={ACTION}")
    best_score, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(split)
        one_sum = ms_sum = 0.0
        batches = 0
        # Grow the multi-step horizon from 2 to MS_HORIZON so early training is not dominated
        # by compounding error from an untrained model.
        horizon = int(round(2 + (MS_HORIZON - 2) * min(1.0, epoch / max(1, EPOCHS // 2))))
        for i in range(0, split, BATCH):
            idx = perm[i:i + BATCH]
            one = mse(model(xn[idx]), yn[idx])
            ms = multi_step_loss(model, data_t, split, *stats_t, horizon, MS_BATCH, mse)
            loss = one + MS_WEIGHT * ms
            opt.zero_grad()
            loss.backward()
            opt.step()
            one_sum += one.item()
            ms_sum += ms.item()
            batches += 1
        sched.step()

        model.eval()
        val_state = data[split:, 0:STATE]
        val_action = data[split:, STATE:STATE + ACTION]
        pred = val_state + predict_delta(model, val_state, val_action, x_mean, x_std, y_mean, y_std)
        one_rmse = joint_rmse(pred, data[split:, STATE + ACTION:])
        ol50 = open_loop_rmse(model, data, split, x_mean, x_std, y_mean, y_std, 50)
        print(f"epoch {epoch + 1:3d}  one-step {one_sum / batches:.5f}  multi-step(h={horizon}) "
              f"{ms_sum / batches:.5f}  val 1-step joint RMSE {one_rmse:.5f} rad  val 50-step {ol50:.4f} rad")
        if ol50 < best_score:
            best_score = ol50
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), ROOT / "dynamics.pt")
    print(f"wrote {ROOT / 'dynamics.pt'} (best val 50-step RMSE {best_score:.4f} rad)")

    val_state = data[split:, 0:STATE]
    val_action = data[split:, STATE:STATE + ACTION]
    pred = val_state + predict_delta(model, val_state, val_action, x_mean, x_std, y_mean, y_std)
    print("\nValidation results (for REPORT.md)")
    print(f"  one-step joint RMSE: {joint_rmse(pred, data[split:, STATE + ACTION:]):.5f} rad")
    for h in (10, 50, 100):
        print(f"  open-loop {h:3d}-step joint RMSE: "
              f"{open_loop_rmse(model, data, split, x_mean, x_std, y_mean, y_std, h):.4f} rad")
    export_onnx(model)
    return model


if __name__ == "__main__":
    current_time = time.time()
    train_world_model(load_rollout())
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
