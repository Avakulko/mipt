import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Asian_GBM_discrete_TEST4 import asian_call as ACD4
from Asian_GBM_discrete import asian_call as ACD6
from MonteCarlo import MC_asian_LNMR

if __name__ == '__main__':

    t0 = pd.to_datetime('2015-04-23')
    # t0 = pd.to_datetime('2000-12-18')
    temp = pd.read_csv("Data/DCOILWTICO.csv", index_col='DATE', parse_dates=True)
    S0 = float(temp.loc[t0]['DCOILWTICO'])
    K = 20
    T = 1
    r = 0.05
    delta = 0
    nsim = 10000
    N = 252
    sigmas = np.linspace(0.001, 1, num=10)
    MCs = list()
    acd4s = list()
    acd6s = list()
    for sigma in sigmas:
        acd4 = ACD4(N, S0, K, sigma, T, r, delta)
        acd6 = ACD6(N, S0, K, sigma, T, r, delta)
        # MC = 0
        MC = MC_asian_LNMR(t0=t0, S0=S0, K=K, sigma=sigma, T=T, r=r, alpha=0, nsim=nsim, N=N)
        print(f'Sigma: {sigma}    Approx4: {acd4}    Approx6: {acd6}    MC: {MC}')
        MCs.append(MC)
        acd4s.append(acd4)
        acd6s.append(acd6)

    plt.plot(sigmas, MCs, label='MC')
    plt.plot(sigmas, acd4s, color='r', label='z^4')
    plt.plot(sigmas, acd6s, color='g', label='z^6')
    plt.legend()
    plt.show()
