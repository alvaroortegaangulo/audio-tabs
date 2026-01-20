from __future__ import annotations
import numpy as np

def viterbi_decode(emissions: np.ndarray, switch_penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """
    emissions: [states, frames] probs (suman 1 por frame)
    Minimiza coste = -log(p_emit) + penalty(si cambia de estado).
    """
    states, frames = emissions.shape
    logp = -np.log(np.clip(emissions, 1e-9, 1.0)).astype(np.float32)

    dp = np.zeros((states, frames), dtype=np.float32)
    back = np.zeros((states, frames), dtype=np.int32)

    dp[:, 0] = logp[:, 0]
    back[:, 0] = 0

    for t in range(1, frames):
        prev = dp[:, t - 1]
        for s in range(states):
            # quedarse
            best_cost = prev[s]
            best_k = s
            # cambiar: prev[k] + switch_penalty
            # (vectorizado)
            costs = prev + switch_penalty
            costs[s] = prev[s]
            k = int(np.argmin(costs))
            best_cost = float(costs[k])
            best_k = k

            dp[s, t] = best_cost + logp[s, t]
            back[s, t] = best_k

    last = int(np.argmin(dp[:, -1]))
    path = np.zeros(frames, dtype=np.int32)
    path[-1] = last
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    conf = emissions[path, np.arange(frames)].astype(np.float32)
    return path, conf
