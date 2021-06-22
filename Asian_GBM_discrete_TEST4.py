import numpy as np

np.random.seed(422)
from scipy.stats import norm
import time


def asian_call(N, S0, K, sigma, T, r, delta):
    def a1(z):
        return -(z ** 2 * dU2_0) / (2 * U2_0)

    def a2(z):
        return 2 * a1(z) ** 2 - (z ** 4 * ddU2_0) / (2 * U2_0)

    def a3(z):
        return 0

    def b1(z):
        return z ** 4 / (4 * U1 ** 3) * E["A'2(0)A''(0)"]

    def b2(z):
        return a1(z) ** 2 - 1 / 2 * a2(z)

    def c1(z):
        return 0

    # U1=A(0)
    def c2(z):
        return 0

    # U1=A(0)
    def c3(z):
        return 0

    def c4(z):
        return 0

    # Не участвует в расчетах
    # d1(1) - d2(1) + d3(1) - d4(1) == 0
    def d1(z):
        return 1 / 2 * (6 * a1(z) ** 2 + a2(z) - 4 * b1(z) + 2 * b2(z)) - 1 / 6 * (
                120 * a1(z) ** 3 - a3(z) + 6 * (24 * c1(z) - 6 * c2(z) + 2 * c3(z) - c4(z)))

    def d2(z):
        return 1 / 2 * (10 * a1(z) ** 2 + a2(z) - 6 * b1(z) + 2 * b2(z)) - (
                128 * a1(z) ** 3 / 3 - a3(z) / 6 + 2 * a1(z) * b1(z) - a1(z) * b2(z) + 50 * c1(z) - 11 * c2(
            z) + 3 * c3(z) - c4(z))

    def d3(z):
        return (2 * a1(z) ** 2 - b1(z)) - 1 / 3 * (
                88 * a1(z) ** 3 + 3 * a1(z) * (5 * b1(z) - 2 * b2(z)) + 3 * (35 * c1(z) - 6 * c2(z) + c3(z)))

    def d4(z):
        return -20 * a1(z) ** 3 / 3 + a1(z) * (-4 * b1(z) + b2(z)) - 10 * c1(z) + c2(z)

    def p(y):
        return norm.pdf(y, loc=m_1, scale=np.sqrt(v_1))

    def dp(y):
        return p(y) * (m_1 - y) / v_1

    def ddp(y):
        return (dp(y) * (m_1 - y) - p(y)) / v_1

    g = r - delta

    chi = 1 / N
    # noinspection PyUnresolvedReferences
    t = np.linspace(start=0, stop=T, num=N)
    # t, dt = np.linspace(start=0, stop=T, num=N, endpoint=False, retstep=True)
    # t += dt
    S_bar = chi * S0 * np.exp(g * t).reshape(1, -1)
    i = np.arange(N).reshape(-1, 1)
    j = i.T
    rho_bar = sigma ** 2 * t[np.minimum(i, j)]

    U1 = np.sum(S_bar)
    U2_1 = np.sum(S_bar * S_bar.T * np.exp(rho_bar))
    U2_0 = np.sum(S_bar * S_bar.T)
    dU2_0 = np.sum(S_bar * S_bar.T * rho_bar)
    ddU2_0 = np.sum(S_bar * S_bar.T * rho_bar ** 2)
    dddU2_0 = 0

    # 2.2 The Efficiency of the Approximation
    A_bar = np.sum(S_bar * rho_bar, axis=1)
    rho_star = np.sqrt(S_bar) * np.sqrt(S_bar.T) * rho_bar
    E = {
        "A'2(0)A''(0)": 2 * np.sum(S_bar * A_bar ** 2),
        "A'2(0)A''2(0)": 0,
        "A'3(0)A(3)(0)": 0,
        "A'(0)A''(0)A(3)(0)": 0,
        "A''3(0)": 0
    }

    z1 = d2(1) - d3(1) + d4(1)
    z2 = d3(1) - d4(1)
    z3 = d4(1)

    m_1 = 2 * np.log(U1) - 0.5 * np.log(U2_1)
    v_1 = np.log(U2_1) - 2 * np.log(U1)  # Дисперсия
    y = np.log(K)
    y1 = (m_1 - y) / np.sqrt(v_1) + np.sqrt(v_1)
    y2 = y1 - np.sqrt(v_1)

    BC = (U1 * np.exp(-r * T) * norm.cdf(y1) - K * np.exp(-r * T) * norm.cdf(y2)) + np.exp(-r * T) * K * (
                z1 * p(y) + z2 * dp(y) + z3 * ddp(y))

    return BC


if __name__ == '__main__':

    # S0 = 100
    # sigma = 0.2
    # sigmas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # K = 100
    # Ks = [95, 100, 105]
    # T = 3
    # Ts = [1, 2, 3, 4, 5]
    # r = 0.09
    # delta = 0
    # N = 252  # 252 дня в году. Ежедневное усреднение
    # N = int(np.ceil(T * 365 / 7))  # 365 дней в году. Усреднение по неделям
    #
    # # Table 4
    # # Pricing Asian and Basket Options Via Taylor Expansion
    # Cs = list()
    # MCs = list()
    # for sigma in sigmas:
    #     for K in Ks:
    #         start = time.time()
    #         print((sigma, K), round(asian_call(S0, K, sigma, T, r, delta), 4))

    # То же самое но до 6 порядка
    # (0.05, 95) 15.1197
    # (0.05, 100) 11.3069
    # (0.05, 105) 7.5562
    # (0.1, 95) 15.2165
    # (0.1, 100) 11.6394
    # (0.1, 105) 8.3913
    # (0.2, 95) 16.6365
    # (0.2, 100) 13.7634
    # (0.2, 105) 11.2134
    # (0.3, 95) 19.0179
    # (0.3, 100) 16.5755
    # (0.3, 105) 14.3774
    # (0.4, 95) 21.7307
    # (0.4, 100) 19.569
    # (0.4, 105) 17.5978
    # (0.5, 95) 24.5583
    # (0.5, 100) 22.6032
    # (0.5, 105) 20.8023

    import pandas as pd
    from MonteCarlo import MC_asian_LNMR

    t0 = pd.to_datetime('2015-04-23')
    # t0 = pd.to_datetime('2000-12-18')
    temp = pd.read_csv("Data/DCOILWTICO.csv", index_col='DATE', parse_dates=True)
    S0 = float(temp.loc[t0]['DCOILWTICO'])
    sigma = 0.4
    K = 40
    T = 2
    r = 0.05
    delta = 0
    N = 252
    AC = asian_call(N, S0, K, sigma, T, r, delta)
    print(AC)
