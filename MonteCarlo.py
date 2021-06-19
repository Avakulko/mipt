import numpy as np

np.random.seed(123)

import matplotlib.pyplot as plt


def MC_asian_LNMR(S0, K, sigma, T, r, alpha, nsim=10000, N=252):
    dt = T / N
    t = dt * np.arange(N + 1)
    x0 = np.log(S0)
    vp = list()
    ap = list()
    theta = r + alpha * np.log(S0) + alpha * r * t + sigma ** 2 / 4 * (
            1 - np.exp(-2 * alpha * t)) - 0.5 * sigma ** 2

    for _ in range(nsim):
        x = x0 * np.ones(N)  # x_t = ln(S(t))
        for i in range(N - 1):
            x[i + 1] = x[i] + (theta[i] - alpha * x[i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        # plt.plot(t, np.exp(x))
        vp.append(x[-1])  # vanilla payoff
        ap.append(np.maximum(np.mean(np.exp(x)) - K, 0))  # asian option payoff

    # plt.show()
    vanilla_call = np.mean(np.maximum(np.exp(-r * T) * (np.exp(vp) - K), 0))
    # print(f'Vanilla call: {vC}')
    asian_call = np.exp(-r * T) * np.mean(ap)
    # print(f'Asian call: {aC}')
    return asian_call


if __name__ == '__main__':

    S0 = 100
    T = 1
    K = 100
    r = 0.05
    alpha = 0.5 * 0
    sigma = 0.5
    nsim = 1000
    N = 252
    print(MC_asian_LNMR(S0, K, sigma, T, r, alpha, nsim=10000, N=252))
    pass
