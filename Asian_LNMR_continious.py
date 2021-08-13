import numpy as np
from scipy.stats import norm


def asian_call(N, S0, K, sigma, alpha, T, r):
    def a1(z):
        return -(z ** 2 * dU2_0) / (2 * U2_0)

    def a2(z):
        return 2 * a1(z) ** 2 - (z ** 4 * ddU2_0) / (2 * U2_0)

    def b1(z):
        return z ** 4 / (4 * U1 ** 3) * E["A'2(0)A''(0)"]

    def b2(z):
        return a1(z) ** 2 - 1 / 2 * a2(z)

    def d2(z):
        return 1 / 2 * (10 * a1(z) ** 2 + a2(z) - 6 * b1(z) + 2 * b2(z))

    def d3(z):
        return 2 * a1(z) ** 2 - b1(z)

    def p(y):
        return norm.pdf(y, loc=m_1, scale=np.sqrt(v_1))

    def dp(y):
        return p(y) * (m_1 - y) / v_1

    t = np.linspace(start=0, stop=T, num=N).reshape(-1, 1)  # len(t) должна быть N, а не T*N, как в Монте-Карло. Иначе не работает при T>1
    # Интеграл для U_2(1) не берется, поэтому считаем его численно
    U2_1 = S0 ** 2 / (N ** 2) * np.sum(np.exp((sigma ** 2) / (2 * alpha) * (np.exp(-alpha * (t - t.T)) - np.exp(-alpha * (t + t.T)))))

    U1 = S0
    U2_0 = S0**2
    dU2_0 = S0**2 * (sigma**2 / (2*alpha)) * ((4*np.exp(-alpha*T) - np.exp(-2*alpha*T) - 3)/(alpha**2 * T**2) + 2/(alpha*T))
    ddU2_0 = S0**2 * (sigma**4 / (4*alpha**2)) * (4*alpha*T + 8*alpha*np.exp(-2*alpha*T)*T + 4*np.exp(-2*alpha*T) + np.exp(-4*alpha*T) - 5) / (4 * alpha**2 * T**2)

    E = {
        "A'2(0)A''(0)": S0**3 * (sigma**4 / alpha**2) * (8*alpha*T + 8*np.exp(-alpha*T)*alpha*T - 4*np.exp(-2*alpha*T)*alpha*T + 4*np.exp(-3*alpha*T) + 28*np.exp(-alpha*T) - np.exp(-4*alpha*T) - 12*np.exp(-2*alpha*T) -19) / (4 * alpha**3 * T**3),
    }

    z1 = d2(1) - d3(1)
    z2 = d3(1)

    m_1 = 2 * np.log(U1) - 0.5 * np.log(U2_1)
    v_1 = np.log(U2_1) - 2 * np.log(U1)  # Дисперсия
    y = np.log(K)
    y1 = (m_1 - y) / np.sqrt(v_1) + np.sqrt(v_1)
    y2 = y1 - np.sqrt(v_1)

    BC = (U1 * np.exp(-r * T) * norm.cdf(y1) - K * np.exp(-r * T) * norm.cdf(y2)) + np.exp(-r * T) * K * (
            z1 * p(y) + z2 * dp(y))

    return BC


if __name__ == '__main__':

    import pandas as pd
    from MonteCarlo import MC_asian_LNMR

    t0 = pd.to_datetime('2015-04-23')
    # t0 = pd.to_datetime('2018-04-23')
    # t0 = pd.to_datetime('2000-12-18')
    temp = pd.read_csv("Data/DCOILWTICO.csv", index_col='DATE', parse_dates=True)
    S0 = float(temp.loc[t0]['DCOILWTICO'])
    sigma = 0.2
    K = 40
    T = 1  # При больших T может случиться что отсутствуют форвардные цены. Тогда интерполяция кривой вернет nan
    r = 0.05
    N = 252
    alpha = 0.5
    AC = asian_call(N, S0, K, sigma, alpha, T, r)
    print(f'Approximation = {AC}')
    MC = MC_asian_LNMR(t0, S0, K, sigma, alpha, T, r, nsim=10000, N=252)
    print(f'Monte-Carlo = {MC}')
