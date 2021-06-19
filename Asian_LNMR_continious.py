import numpy as np

np.random.seed(422)
from scipy.stats import norm
import time
from forecasting_metrics import mae, mape


def asian_call(S0, K, sigma, alpha, T, r, delta, averaging='discrete'):
    def a1(z):
        return -(z ** 2 * dU2_0) / (2 * U2_0)

    def a2(z):
        return 2 * a1(z) ** 2 - (z ** 4 * ddU2_0) / (2 * U2_0)

    def a3(z):
        return 6 * a1(z) * a2(z) - 4 * a1(z) ** 3 - (z ** 6 * dddU2_0) / (2 * U2_0)

    def b1(z):
        return z ** 4 / (4 * U1 ** 3) * E["A'2(0)A''(0)"]

    # осторожно с матожами
    def b2(z):
        return a1(z) ** 2 - 1 / 2 * a2(z)

    # осторожно с матожами
    def c1(z):
        return -a1(z) * b1(z)

    # U1=A(0)
    def c2(z):
        return z ** 6 / (144 * U1 ** 4) * (9 * E["A'2(0)A''2(0)"] + 4 * E["A'3(0)A(3)(0)"])

    # U1=A(0)
    def c3(z):
        return z ** 6 / (48 * U1 ** 3) * (4 * E["A'(0)A''(0)A(3)(0)"] + E["A''3(0)"])

    # осторожно с матожами
    def c4(z):
        return a1(z) * a2(z) - 2 / 3 * a1(z) ** 3 - 1 / 6 * a3(z)

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

    if averaging == 'discrete':
        N = int(np.ceil(T * 365 / 7))
        # noinspection PyUnresolvedReferences
        t = np.linspace(start=0, stop=T, num=N)

        # t, dt = np.linspace(start=0, stop=T, num=N, endpoint=False, retstep=True)
        # t += dt

        def U2(z):
            x = np.sum(np.exp((z ** 2 * sigma ** 2) / (2 * alpha) * (1 - np.exp(-2 * alpha * t))))
            y = 0
            for i in range(N - 1):
                for j in range(i + 1, N):
                    y += np.exp((z ** 2 * sigma ** 2) / (2 * alpha) * (
                                np.exp(-alpha * (t[j] - t[i])) - np.exp(-alpha * (t[j] + t[i]))))
            return S0 ** 2 / N ** 2 * (x + 2 * y)

        U1 = S0
        U2_0 = 0
        x = 0
        y = 0
        x += S0 ** 2 / N
        for i in range(N - 1):
            for j in range(i + 1, N):
                y += 2 * S0 ** 2 / N ** 2
        U2_0 = x + y
        return

        # U2_1 =
        # U2_0 =
        # dU2_0 =
        # ddU2_0 =
        # dddU2_0 =

        # 2.2 The Efficiency of the Approximation
        A_bar = np.sum(S_bar * rho_bar, axis=1)
        rho_star = np.sqrt(S_bar) * np.sqrt(S_bar.T) * rho_bar
        E = {
            "A'2(0)A''(0)": 2 * np.sum(S_bar * A_bar ** 2),
            "A'2(0)A''2(0)": 8 * np.sum(A_bar * S_bar * rho_bar * (S_bar * A_bar).T) + 2 * dU2_0 * ddU2_0,
            "A'3(0)A(3)(0)": 6 * np.sum(S_bar * A_bar ** 3),
            "A'(0)A''(0)A(3)(0)": 6 * np.sum(S_bar * rho_bar ** 2 * (S_bar * A_bar).T),
            "A''3(0)": 8 * np.einsum("ij,jk,ki->", rho_star, rho_star, rho_star)
        }

        z1 = d2(1) - d3(1) + d4(1)
        z2 = d3(1) - d4(1)
        z3 = d4(1)
    else:
        pass

    m_1 = 2 * np.log(U1) - 0.5 * np.log(U2_1)
    v_1 = np.log(U2_1) - 2 * np.log(U1)  # Дисперсия
    y = np.log(K)
    y1 = (m_1 - y) / np.sqrt(v_1) + np.sqrt(v_1)
    y2 = y1 - np.sqrt(v_1)

    BC = (U1 * np.exp(-r * T) * norm.cdf(y1) - K * np.exp(-r * T) * norm.cdf(y2)) + np.exp(-r * T) * K * (
            z1 * p(y) + z2 * dp(y) + z3 * ddp(y))

    return BC


if __name__ == '__main__':
    S0 = 100
    sigma = 0.2
    K = 100
    T = 3
    r = 0.09
    delta = 0
    alpha = 1
    asian_call(S0, K, sigma, alpha, T, r, delta, averaging='discrete')