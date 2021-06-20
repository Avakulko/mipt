import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def interp_forward(t0, T, N=252):
    temp = pd.read_csv("Data/DCOILWTICO.csv", index_col='DATE', parse_dates=True)
    S0 = float(temp.loc[t0]['DCOILWTICO'])

    data = [S0]
    for i in range(1, 12 * T + 1):
        temp = pd.read_csv(f"Data/CME_CL{i}.csv", index_col='Date', parse_dates=True)
        data.append(temp.loc[t0]['Last'])

    t1m = np.linspace(start=0, stop=T * N, num=12 * T + 1) / N
    F = interp1d(t1m, data, kind='cubic')
    # plt.scatter(t1m, data)
    return F


if __name__ == '__main__':
    t0 = pd.to_datetime('2015-04-23')
    t0 = pd.to_datetime('2000-12-18')
    T = 1
    N = 252
    F = interp_forward(t0, T, N)

    t = np.linspace(start=0, stop=T, num=T * N)

    lnF = np.log(F(t))
    grad = np.gradient(lnF, t)
    plt.plot(t, F(t))
    plt.plot(t, np.log(F(t)))
    plt.plot(t, grad)
    plt.show()
