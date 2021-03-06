import numpy as np

np.random.seed(422)
from scipy.stats import norm
from MonteCarlo import MC_asian_LNMR

import matplotlib.pyplot as plt


def asian_call(S, K, sigma, T, r, delta):
    def p(y):
        return norm.pdf(y, loc=m_1, scale=np.sqrt(v_1))

    def dp(y):
        return p(y) * (m_1 - y) / v_1

    def ddp(y):
        return (dp(y) * (m_1 - y) - p(y)) / v_1

    g = r - delta

    # В статье U1 и U2 посчитаны с ошибками. Ниже приведены верные формулы
    U1 = S / (g * T) * (np.exp(g * T) - 1)
    U2_1 = 2 * S ** 2 / ((g + sigma ** 2) * T) * (
            (np.exp((2 * g + sigma ** 2) * T) - 1) / ((2 * g + sigma ** 2) * T) - (np.exp(g * T) - 1) / (g * T))

    x = g * T
    z1 = -sigma ** 4 * T ** 2 * (
            1 / 45 + x / 180 - 11 * x ** 2 / 15120 - x ** 3 / 2520 + x ** 4 / 113400) - sigma ** 6 * T ** 3 * (
                 1 / 11340 - 13 * x / 30240 - 17 * x ** 2 / 226800 + 23 * x ** 3 / 453600 + 59 * x ** 4 / 5987520)
    z2 = -sigma ** 4 * T ** 2 * (
            1 / 90 + x / 360 - 11 * x ** 2 / 30240 - x ** 3 / 5040 + x ** 4 / 226800) + sigma ** 6 * T ** 3 * (
                 31 / 22680 + 11 * x / 60480 - 37 * x ** 2 / 151200 - 19 * x ** 3 / 302400 + 953 * x ** 4 / 59875200)
    z3 = sigma ** 6 * T ** 3 * (
            2 / 2835 - x / 60480 - 2 * x ** 2 / 14175 - 17 * x ** 3 / 907200 + 13 * x ** 4 / 1247400)

    m_1 = 2 * np.log(U1) - 0.5 * np.log(U2_1)
    v_1 = np.log(U2_1) - 2 * np.log(U1)  # Дисперсия
    y = np.log(K)
    y1 = (m_1 - y) / np.sqrt(v_1) + np.sqrt(v_1)
    y2 = y1 - np.sqrt(v_1)

    BC = (U1 * np.exp(-r * T) * norm.cdf(y1) - K * np.exp(-r * T) * norm.cdf(y2)) + np.exp(-r * T) * K * (
            z1 * p(y) + z2 * dp(y) + z3 * ddp(y))

    return BC


if __name__ == '__main__':

    # S = 100
    # sigma = 0.2
    # sigmas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # K = 100
    # Ks = [95, 100, 105]
    # T = 3
    # Ts = [1, 2, 3, 4, 5]
    # r = 0.09
    # delta = 0

    # # Table 2
    # # Pricing Asian and Basket Options Via Taylor Expansion
    # for sigma in sigmas:
    #     for K in Ks:
    #         start = time.time()
    #         print((sigma, K), round(asian_call(S, K, sigma, T, r, delta), 4))

    # ##############################
    # import pandas as pd

    # # При alpha = 0 совпадает с Монте-Карло в случае когда F = S0 e^{rT}
    # t0 = pd.to_datetime('2015-04-23')
    # # t0 = pd.to_datetime('2000-12-18')
    # temp = pd.read_csv("Data/DCOILWTICO.csv", index_col='DATE', parse_dates=True)
    # S0 = float(temp.loc[t0]['DCOILWTICO'])
    # T = 1
    # K = 20
    # r = 0.05
    # sigma = 0.5
    # delta = 0
    # print(asian_call(S0, K, sigma, T, r, delta))
    # asian_mc = list()
    # asian_approx = list()
    # for T in range(1, 4):
    #     asian_mc.append(MC_asian_LNMR(t0, S0, K, sigma, T, r, alpha=0, nsim=1000000))
    #     asian_approx.append(asian_call(S0, K, sigma, T, r, delta))
    #
    # plt.plot(asian_mc)
    # plt.plot(asian_approx)
    # plt.show()
    # print(asian_mc)
    # print(asian_approx)

    import pandas as pd
    from MonteCarlo import MC_asian_LNMR

    t0 = pd.to_datetime('2015-04-23')
    # t0 = pd.to_datetime('2000-12-18')
    temp = pd.read_csv("Data/DCOILWTICO.csv", index_col='DATE', parse_dates=True)
    S0 = float(temp.loc[t0]['DCOILWTICO'])
    sigma = 0.3
    K = 20
    T = 1
    r = 0.05
    delta = 0
    N = 252
    AC = asian_call(S0, K, sigma, T, r, delta)
    print(AC)
