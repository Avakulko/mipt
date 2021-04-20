import numpy as np

def call_payoff(S, K):
    return max(S - K, 0)

def A(S: np.array) -> int:
    return np.mean(S)

def simulate_paths():
    N = 100
    T = 1
    dt = T / 365
    S0 = 0
    