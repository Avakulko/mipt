import numpy as np
from scipy.stats import norm
import time

def asian_call(S, K, sigma, T, r, delta, averaging='discrete'):

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
        chi = 1 / N
        t = np.linspace(start=0, stop=T, num=N)

        S_bar = chi * S * np.exp(g * t).reshape(1, -1)
        i = np.arange(N).reshape(-1, 1)
        j = i.T
        rho_bar = sigma ** 2 * t[np.minimum(i, j)]

        U1 = np.sum(S_bar)
        U2_1 = np.sum(S_bar * S_bar.T * np.exp(rho_bar))
        U2_0 = np.sum(S_bar * S_bar.T)
        dU2_0 = np.sum(S_bar * S_bar.T * rho_bar)
        ddU2_0 = np.sum(S_bar * S_bar.T * rho_bar**2)
        dddU2_0 = np.sum(S_bar * S_bar.T * rho_bar**3)

        # 2.2 The Efficiency of the Approximation
        A_bar = np.sum(S_bar*rho_bar, axis=1)
        rho_star = np.sqrt(S_bar) * np.sqrt(S_bar.T) * rho_bar
        E = {
            "A'2(0)A''(0)": 2 * np.sum(S_bar * A_bar ** 2),
            "A'2(0)A''2(0)": 8 * np.sum(A_bar * S_bar * rho_bar * (S_bar * A_bar).T) + 2 * dU2_0 * ddU2_0,
            "A'3(0)A(3)(0)": 6 * np.sum(S_bar * A_bar ** 3),
            "A'(0)A''(0)A(3)(0)": 6 * np.sum(S_bar * rho_bar**2 * (S_bar * A_bar).T),
            "A''3(0)": 8 * np.einsum("ij,jk,ki->", rho_star, rho_star, rho_star)
        }
        
        z1 = d2(1) - d3(1) + d4(1)
        z2 = d3(1) - d4(1)
        z3 = d4(1)

    else:
        # В статье U1 и U2 посчитаны с ошибками. Ниже приведены верные формулы
        U1 = S/(g*T) * (np.exp(g*T) - 1)
        U2_1 = 2*S**2/((g + sigma**2)*T) * ((np.exp((2*g + sigma**2)*T) - 1) / ((2*g + sigma**2)*T) - (np.exp(g*T) - 1)/(g*T))

        x = g * T
        z1 = -sigma**4 * T**2 * (1/45 + x/180 - 11*x**2/15120 - x**3/2520 + x**4/113400) - sigma**6 * T**3 * (1/11340 - 13*x/30240 - 17*x**2/226800 + 23*x**3/453600 + 59*x**4/5987520)
        z2 = -sigma**4 * T**2 * (1/90 + x/360 - 11*x**2/30240 - x**3/5040 + x**4/226800) + sigma**6 * T**3 * (31/22680 + 11*x/60480 - 37*x**2/151200 - 19*x**3/302400 + 953*x**4/59875200)
        z3 = sigma**6 * T**3 * (2/2835 - x/60480 - 2*x**2/14175 - 17*x**3/907200 + 13*x**4/1247400)

    m_1 = 2 * np.log(U1) - 0.5 * np.log(U2_1)
    v_1 = np.log(U2_1) - 2 * np.log(U1)  # Дисперсия
    y = np.log(K)
    y1 = (m_1 - y) / np.sqrt(v_1) + np.sqrt(v_1)
    y2 = y1 - np.sqrt(v_1)

    BC = (U1 * np.exp(-r * T) * norm.cdf(y1) - K * np.exp(-r * T) * norm.cdf(y2)) + np.exp(-r * T) * K * (z1 * p(y) + z2 * dp(y) + z3 * ddp(y))

    return BC


if __name__ == '__main__':

    S = 100
    Ks = [95, 100, 105]
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    T = 3
    r = 0.09
    delta = 0

    # Table 4
    # Pricing Asian and Basket Options Via Taylor Expansion
    for sigma in sigmas:
        for K in Ks:
            start = time.time()
            print((sigma, K), asian_call(S, K, sigma, T, r, delta, averaging='discrete'), f'in {time.time() - start} seconds')
    print()
    # Table 2
    # Pricing Asian and Basket Options Via Taylor Expansion
    for sigma in sigmas:
        for K in Ks:
            start = time.time()
            print((sigma, K), asian_call(S, K, sigma, T, r, delta, averaging='continious'), f'in {time.time() - start} seconds')
            pass