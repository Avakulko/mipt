import numpy as np
from Forward_curve import interp_forward
import pandas as pd

np.random.seed(123)

import matplotlib.pyplot as plt

# При больших T может случиться что отсутствуют форвардные цены. Тогда интерполяция кривой вернет nan
def MC_asian_LNMR(t0, S0, K, sigma, alpha, T, r, nsim=10000, N=252):
    def generate_paths():
        total_paths = 10 * nsim
        x = x0 * np.ones((T * N, total_paths))  # x_t = ln(S(t))
        for i in range(len(t) - 1):
            x[i + 1, :] = x[i, :] + (theta[i] - alpha * x[i, :]) * dt + sigma * np.sqrt(dt) * np.random.normal(
                size=total_paths)
        return x

    dt = 1 / (N - 1)
    t = np.linspace(start=0, stop=T, num=T * N)
    x0 = np.log(S0)
    F = interp_forward(t0, T, N)
    lnF = np.log(F(t))
    dlnFdt = np.gradient(lnF, t)

    # Commodity. Для LNMR
    chi = (S0 / (F(t))).reshape(-1, 1)
    theta = dlnFdt + alpha * lnF + sigma ** 2 / 4 * (1 - np.exp(-2 * alpha * t)) - 0.5 * sigma ** 2

    # Not commodity. Для GBM
    # chi = 1
    # theta = r + alpha * np.log(S0) + alpha * r * t + sigma ** 2 / 4 * (1 - np.exp(-2 * alpha * t)) - 0.5 * sigma ** 2

    paths = generate_paths()
    asian_calls = list()

    for _ in range(20):
        x = paths[:, np.random.randint(paths.shape[1], size=nsim)]
        asian_call = np.exp(-r*T) * np.mean(np.maximum((np.mean(chi * np.exp(x), axis=0) - K), 0))
        asian_calls.append(asian_call)

        # Траектории S(t)
        # plt.plot(t, np.exp(x))
        # plt.show()
        # Проверка совпадения моментов
        # theoretical_means = lnF - sigma ** 2 / (4 * alpha) * (1 - np.exp(-2 * alpha * t))
        # plt.plot(t, theoretical_means)
        # plt.plot(t, np.mean(x, axis=1))
        # theoretical_vars = sigma ** 2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
        # plt.plot(t, theoretical_vars)
        # plt.plot(t, np.var(x, axis=1))
        # theoretical_3rd_central_moment = np.mean((x - np.mean(x, axis=1).reshape(-1, 1))**3, axis=1)
        # plt.plot(t, theoretical_3rd_central_moment)
        # plt.show()

    asian_call = np.mean(asian_calls)
    lower_boundary = np.percentile(asian_calls, 5, interpolation='lower')
    upper_boundary = np.percentile(asian_calls, 95, interpolation='higher')
    CI = upper_boundary - lower_boundary
    print(f'Monte-Carlo CI = [{lower_boundary}, {upper_boundary}]')

    return asian_call


if __name__ == '__main__':
    T = 3
    K = 20
    r = 0.05
    alpha = 0.5
    sigma = 0.5
    nsim = 10
    N = 252
    t0 = pd.to_datetime('2015-04-23')
    # t0 = pd.to_datetime('2000-12-18')
    temp = pd.read_csv("Data/DCOILWTICO.csv", index_col='DATE', parse_dates=True)
    S0 = float(temp.loc[t0]['DCOILWTICO'])
    print(MC_asian_LNMR(t0, S0, K, sigma, alpha, T, r, nsim=nsim, N=N))
    pass
