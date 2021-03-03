#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons

from math import pi


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def main():
    # read data
    #file = "../Spike/data/Firing_rate/lifac_firing_rate.csv"
    file = "../Spike/data/Firing_rate/lifac_two_sigs_firing_rate.csv"
    #file = "../Spike/data/Firing_rate/lif_firing_rate.csv"
    data = np.genfromtxt(file, delimiter=',')
    t = data[1:, 0]
    rate = data[1:, 1]

    rate = running_mean(rate, 10)
    t = t[0:rate.size]

    # analytics
    lifac = neurons.LIFAC(1.1, 1e-3, 1e-2, 10.0)
    alpha = 0.03
    f1 = 0.26
    beta = 0.03
    f2 = 0.1
    rate_ana_lif = neurons.firing_rate_signal_nonlinear_two_sigs(lifac.lif, t, alpha, f1, beta, f2)
    rate_ana_lifac = neurons.firing_rate_signal_nonlinear_two_sigs(lifac, t, alpha, f1, beta, f2)
    rate_ana_lifac_lin = neurons.firing_rate_signal_linear_two_sigs(lifac, t, alpha, f1, beta, f2)
    #alpha = 0.05
    #f = 0.1834
    #rate_ana_lif = neurons.firing_rate_signal_nonlinear(lifac.lif, t, alpha, f)
    #rate_ana_lifac = neurons.firing_rate_signal_nonlinear(lifac, t, alpha, f)


    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$r(t)$")
    ax.set_xlim(70, 100)

    ax.plot(t, rate)
    ax.plot(t, rate_ana_lif)
    ax.plot(t, rate_ana_lifac)
    ax.plot(t, rate_ana_lifac_lin)

    plt.show()


if __name__ == "__main__":
    main()
