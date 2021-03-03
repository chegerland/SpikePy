#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons

from math import pi


def main():
    # define LIFAC and LIF
    mu = 1.1
    D = 1e-3
    lif = neurons.LIF(mu, D)

    # calculate susceptibilities of LIF
    f = np.logspace(-3, 2.0, num=400)
    chi_1 = np.zeros(shape=(len(f)), dtype=np.complex)
    for i in range(len(f)):
        chi_1[i] = lif.susceptibility_1(2. * pi * f[i])

    # prepare plots
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 24
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    ax1.set_ylabel("$ |\\chi_1(f)| $")
    ax2.set_ylabel("$ \\phi[\\chi_1(f)] $")
    ax1.set_xlabel("$ f $")
    ax2.set_xlabel("$ f $")
    ax1.set_xlim(2e-3, 1e1)
    ax2.set_xlim(2e-3, 1e1)

    # read data
    data = np.genfromtxt("../Spike/data/LIF_Novikov/lif_suscept_novikov.csv", delimiter=',', dtype=np.complex128)
    #data = np.genfromtxt("../Spike/data/LIF_Novikov/lif_suscept_lin.csv", delimiter=',', dtype=np.complex128)
    f_data = data[1:, 0]

    # plot the theory lines
    ax1.semilogx(f_data, np.abs(data[1:, 1]), label="$c = 0.01$")
    ax2.semilogx(f_data, np.angle(data[1:, 1]), label="$c = 0.01$")
    ax1.semilogx(f_data, np.abs(data[1:, 2]), label="$c = 0.1$")
    ax2.semilogx(f_data, np.angle(data[1:, 2]), label="$c = 0.1$")
    ax1.semilogx(f_data, np.abs(data[1:, 3]), label="$c = 0.5$")
    ax2.semilogx(f_data, np.angle(data[1:, 3]), label="$c = 0.5$")
    #ax1.semilogx(f_data, np.abs(data[1:, 4]), label="$c = 0.9$")
    #ax2.semilogx(f_data, np.angle(data[1:, 4]), label="$c = 0.9$")
    ax1.semilogx(f_data, np.abs(data[1:, 5]), label="$c = 1$")
    ax2.semilogx(f_data, np.angle(data[1:, 5]), label="$c = 1$")
    ax1.semilogx(f, np.abs(chi_1), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    ax2.semilogx(f, -np.angle(chi_1), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    ax1.legend(loc="upper left", frameon=False)
    #ax2.legend(loc="lower left", frameon=False)

    plt.subplots_adjust(hspace=0.0)
    # export plot
    fig.savefig("img/suscept_lif_lin_novikov.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
