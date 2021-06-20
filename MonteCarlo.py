import numpy as np
from Forward_curve import interp_forward
import pandas as pd

np.random.seed(123)

import matplotlib.pyplot as plt


def MC_asian_LNMR(t0, S0, K, sigma, T, r, alpha, nsim=10000, N=252):
    dt = 1 / (N - 1)
    t = np.linspace(start=0, stop=T, num=T * N)
    x0 = np.log(S0)
    ap = list()
    F = interp_forward(t0, T, N)
    lnF = np.log(F(t))
    dlnFdt = np.gradient(lnF, t)

    # Commodity
    theta = dlnFdt + alpha * lnF + sigma ** 2 / 4 * (
            1 - np.exp(-2 * alpha * t)) - 0.5 * sigma ** 2

    # Not commodity
    # theta = r + alpha * np.log(S0) + alpha * r * t + sigma ** 2 / 4 * (
    #         1 - np.exp(-2 * alpha * t)) - 0.5 * sigma ** 2

    x = x0 * np.ones((T * N, nsim))  # x_t = ln(S(t))
    for i in range(len(t) - 1):
        x[i + 1, :] = x[i, :] + (theta[i] - alpha * x[i, :]) * dt + sigma * np.sqrt(dt) * np.random.normal(size=nsim)
    # plt.plot(t, np.exp(x))
    ap.append(np.maximum(np.mean(np.exp(x)) - K, 0))  # asian option payoff

    # Проверка совпадения моментов
    # theoretical_means = lnF - sigma ** 2 / (4 * alpha) * (1 - np.exp(-2 * alpha * t))
    # theoretical_vars = sigma ** 2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
    # theoretical_3rd_central_moment = np.mean((x - np.mean(x, axis=1).reshape(-1, 1))**3, axis=1)
    # plt.plot(t, theoretical_means)
    # plt.plot(t, np.mean(x, axis=1))
    # plt.show()
    # plt.plot(t, theoretical_vars)
    # plt.plot(t, np.var(x, axis=1))
    # plt.show()
    # plt.plot(t, theoretical_3rd_central_moment)
    # plt.show()

    # vanilla_call = np.exp(-r * T) * np.mean(np.maximum(np.exp(x[-1, :]) - K, 0))
    # print(f'Vanilla call: {vanilla_call}')
    asian_call = np.exp(-r*T) * np.mean(np.maximum((np.mean(np.exp(x), axis=0) - K), 0))
    # print(f'Asian call: {asian_call}')
    return asian_call

if __name__ == '__main__':
    T = 3
    K = 20
    r = 0.05
    alpha = 0.5 * 0
    sigma = 0.5
    nsim = 100000
    N = 252
    t0 = pd.to_datetime('2015-04-23')
    # t0 = pd.to_datetime('2000-12-18')
    temp = pd.read_csv("Data/DCOILWTICO.csv", index_col='DATE', parse_dates=True)
    S0 = float(temp.loc[t0]['DCOILWTICO'])
    print(MC_asian_LNMR(t0, S0, K, sigma, T, r, alpha, nsim=nsim, N=N))
    pass
