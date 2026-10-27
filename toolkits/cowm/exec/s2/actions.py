"""Shared S2 action sequences: identical inputs for target LIBERO and the Code WM executor."""
import numpy as np


def _hold(a, n):
    return np.tile(np.asarray(a, dtype=np.float64), (n, 1))


def make_sequences():
    """Return {name: [T, 7] actions in [-1, 1]}. Gripper is dim 6."""
    zero = np.zeros(7)
    seqs = {"zero": _hold(zero, 20)}
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
    for d, name in enumerate(names):
        for sign in (+1, -1):
            a = zero.copy()
            a[d] = 0.5 * sign
            # move 10 steps, then settle 10 steps with zero action
            seqs[f"{name}{'+' if sign > 0 else '-'}"] = np.concatenate([_hold(a, 10), _hold(zero, 10)])
    close, open_ = zero.copy(), zero.copy()
    close[6], open_[6] = 1.0, -1.0
    seqs["grip_close_open"] = np.concatenate([_hold(close, 15), _hold(open_, 15)])
    for seed in range(3):
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-0.3, 0.3, size=(60, 6))
        # smooth with a short moving average so motions stay small and continuous
        k = np.ones(5) / 5
        smooth = np.stack([np.convolve(raw[:, j], k, mode="same") for j in range(6)], axis=1)
        grip = np.where(np.arange(60) < 30, -1.0, 1.0)[:, None]
        seqs[f"rand{seed}"] = np.concatenate([smooth, grip], axis=1)
    return seqs
