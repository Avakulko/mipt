import numpy as np
from scipy.stats import norm
import time
import functools

start = time.time()

# Need only m(1)?
def m(zz):
    return 2 * np.log(U1) - 0.5 * np.log(U2(zz))


# Need only v(1)?
# v- дисперсия
def v(zz):
    return np.log(U2(zz)) - 2 * np.log(U1)


@functools.lru_cache(maxsize=None)
def U2(zz):
    return sum([S_bar[i] * S_bar[j] * np.exp(zz * rho_bar[i, j]) for i in range(N) for j in range(N)])


def a1(z):
    return -(z**2 * dU2_0) / (2*U2_0)


def a2(z):
    return 2*a1(z)**2 - (z**4 * ddU2_0) / (2*U2_0)


def a3(z):
    return 6*a1(z)*a2(z) - 4*a1(z)**3 - (z**6 * dddU2_0) / (2*U2_0)


def b1(z):
    return z**4 / (4*U1**3) * E["A'2(0)A''(0)"]


# осторожно с матожами
def b2(z):
    return a1(z)**2 - 1/2 * a2(z)


# осторожно с матожами
def c1(z):
    return -a1(z) * b1(z)


# U1=A(0)
def c2(z):
    return z**6 / (144*U1**4) * (9*E["A'2(0)A''2(0)"] + 4*E["A'3(0)A(3)(0)"])


# U1=A(0)
def c3(z):
    return z**6 / (48*U1**3) * (4*E["A'(0)A''(0)A(3)(0)"] + E["A''3(0)"])


# осторожно с матожами
def c4(z):
    return a1(z)*a2(z) - 2/3 * a1(z)**3 - 1/6 * a3(z)


# Не участвует в расчетах
# d1(1) - d2(1) + d3(1) - d4(1) == 0
def d1(z):
    return 1/2 * (6*a1(z)**2 + a2(z) - 4*b1(z) + 2*b2(z)) - 1/6 * (120*a1(z)**3 - a3(z) + 6 * (24*c1(z) - 6*c2(z) + 2*c3(z) - c4(z)))


def d2(z):
    return 1/2 * (10*a1(z)**2 + a2(z) - 6*b1(z) + 2*b2(z)) - (128*a1(z)**3 / 3 - a3(z)/6 + 2*a1(z)*b1(z) - a1(z)*b2(z) + 50*c1(z) - 11*c2(z) + 3*c3(z) - c4(z))


def d3(z):
    return (2*a1(z)**2 - b1(z)) - 1/3 * (88*a1(z)**3 + 3*a1(z) * (5*b1(z) - 2*b2(z)) + 3*(35*c1(z) - 6*c2(z) + c3(z)))


def d4(z):
    return (-20*a1(z)**3 / 3 + a1(z) * (-4*b1(z) + b2(z)) - 10*c1(z) + c2(z))


def p(y):
    return norm.pdf(y, loc=m(1), scale=np.sqrt(v(1)))


def dp(y):
    return p(y) * (m(1) - y) / v(1)


def ddp(y):
    return (dp(y) * (m(1) - y) - p(y)) / v(1)



r = 0.09
delta = 0
g = r - delta
T = 3
N = int(np.ceil(T * 365 / 7))
Delta = T / (N - 1)
t = np.linspace(start=0, stop=T, num=N)
chi = 1 / N
S = 100
sigma = 0.5
K = 100

S_bar = chi * S * np.exp(g * t)

U1 = np.sum(S_bar)

rho_bar = sigma**2 * np.array([t[min(i, j)] for i in range(N) for j in range(N)]).reshape((N, N))

U2_0 = sum([S_bar[i] * S_bar[j] for i in range(N) for j in range(N)])
dU2_0 = sum([S_bar[i] * S_bar[j] * rho_bar[i, j] for i in range(N) for j in range(N)])
ddU2_0 = sum([S_bar[i] * S_bar[j] * rho_bar[i, j]**2 for i in range(N) for j in range(N)])
dddU2_0 = sum([S_bar[i] * S_bar[j] * rho_bar[i, j]**3 for i in range(N) for j in range(N)])


# 2.2 The Efficiency of the Approximation
A_bar = np.multiply(rho_bar, S_bar[:, np.newaxis]).sum(axis=0)
rho_star = np.array([np.sqrt(S_bar[i]) * rho_bar[i, j] * np.sqrt(S_bar[j]) for i in range(N) for j in range(N)]).reshape((N, N))

E = {
    "A'2(0)A''(0)": 2 * np.sum(S_bar * A_bar**2),
    "A'2(0)A''2(0)": 8 * sum([A_bar[k] * S_bar[k] * rho_bar[k, l] * S_bar[l] * A_bar[l] for k in range(N) for l in range(N)]) + 2*dU2_0*ddU2_0,
    "A'3(0)A(3)(0)": 6 * np.sum(S_bar * A_bar**3),
    "A'(0)A''(0)A(3)(0)": 6 * sum([S_bar[j] * rho_bar[j, k]**2 * S_bar[k] * A_bar[k] for j in range(N) for k in range(N)]),
    "A''3(0)": 8 * sum([rho_star[i, j] * rho_star[j, k] * rho_star[k, i] for i in range(N) for j in range(N) for k in range(N)])
}

z1 = d2(1) - d3(1) + d4(1)
z2 = d3(1) - d4(1)
z3 = d4(1)

y = np.log(K)
y1 = (m(1) - y) / np.sqrt(v(1)) + np.sqrt(v(1))
y2 = y1 - np.sqrt(v(1))

BC = (U1 * np.exp(-r * T) * norm.cdf(y1) - K * np.exp(-r * T) * norm.cdf(y2)) + np.exp(-r * T) * K * (z1 * p(y) + z2 * dp(y) + z3 * ddp(y))
end = time.time()
print(f"Executed in {round(end - start, 1)} seconds")
pass