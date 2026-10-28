"""Minimal CMA-ES (Hansen's (mu/mu_w, lambda) with rank-mu + rank-one updates) on a box, numpy only.

Parameters are optimized in normalized coordinates z in [0, 1]^n (z -> lo + z * (hi - lo)); candidates are
clipped to the box. Small n (1-4) and small budgets are the intended use (S6 dynamics parameters).
"""
import numpy as np


def cmaes(f, x0, sigma0=0.25, popsize=None, generations=8, seed=0, f_batch=None):
    """f(list_of_z) -> list of losses if f_batch is True-ish, else f(z) -> loss. Returns (best_z, best_f, log)."""
    rng = np.random.default_rng(seed)
    n = len(x0)
    lam = popsize or (4 + int(3 * np.log(n)))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    m, sigma = np.array(x0, float), sigma0
    C, pc, ps = np.eye(n), np.zeros(n), np.zeros(n)
    best_z, best_f, log = m.copy(), np.inf, []
    for g in range(generations):
        evals, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(evals, 1e-20))
        arz = rng.standard_normal((lam, n))
        ary = arz @ np.diag(D) @ B.T
        X = np.clip(m + sigma * ary, 0.0, 1.0)
        F = np.asarray(f_batch(list(X)) if f_batch else [f(x) for x in X], float)
        idx = np.argsort(F)
        if F[idx[0]] < best_f:
            best_f, best_z = float(F[idx[0]]), X[idx[0]].copy()
        log.append({"gen": g, "mean": m.tolist(), "sigma": float(sigma), "best": float(F[idx[0]]),
                    "pop": [[*X[i].tolist(), float(F[i])] for i in range(lam)]})
        y = (X[idx[:mu]] - m) / sigma
        m = m + sigma * (w @ y)
        invsqrtC = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (w @ y)
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chin < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (w @ y)
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
            + cmu * (y.T @ np.diag(w) @ y)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))
        m = np.clip(m, 0.0, 1.0)
    return best_z, best_f, log
