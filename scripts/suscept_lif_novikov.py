#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from spike import LIF

from math import pi


def main():
    # define LIFAC and LIF
    mu = 1.1
    D = 1e-3
    lif = LIF(mu, D)

    print("loading data")
    # read data
    data = np.genfromtxt("../Spike/data/LIF_Novikov/lif_suscept_novikov.csv", delimiter=',', dtype=np.complex128)
    f_data = data[1:, 0]

    print("calculating susceptibility")
    # calculate susceptibilities of LIF
    f = np.logspace(-3, 1.5, num=100)
    chi_1 = np.zeros(shape=(len(f)), dtype=complex)
    chi_2 = np.zeros(shape=(len(f)), dtype=complex)
    for i in range(len(f)):
        print("Step", i, "of", len(f))
        chi_1[i] = lif.susceptibility_1(2. * pi * f[i])
        chi_2[i] = lif.susceptibility_2(2. * pi * f[i], 2. * pi * f[i])

    # prepare plots
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 24
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    ax1.set_ylabel("$ |\\chi_1(f)| $")
    ax2.set_ylabel("$ \\phi[\\chi_1(f)] $")
    ax1.set_xlabel("$ f $")
    ax2.set_xlabel("$ f $")
    #ax1.set_xlim(2e-3, 1e1)
    #ax2.set_xlim(2e-3, 1e1)


    # plot the theory lines
    #ax1.semilogx(f_data, np.abs(data[1:, 1]), label="$c = 0.01$")
    #ax2.semilogx(f_data, np.angle(data[1:, 1]), label="$c = 0.01$")
    #ax1.semilogx(f_data, np.abs(data[1:, 3]), label="$c = 0.1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 3]), label="$c = 0.1$")
    #ax1.semilogx(f_data, np.abs(data[1:, 5]), label="$c = 1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 5]), label="$c = 1$")
    #ax1.semilogx(f, np.abs(chi_1), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    #ax2.semilogx(f, -np.angle(chi_1), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    #ax1.legend(loc="upper left", frameon=False)

    #ax1.semilogx(f_data, np.real(data[1:, 2]), label="$c = 0.01$")
    #ax2.semilogx(f_data, np.angle(data[1:, 2]), label="$c = 0.01$")
    #ax1.semilogx(f_data, np.real(data[1:, 4]), label="$c = 0.1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 4]), label="$c = 0.1$")
    ax1.semilogx(f_data, np.real(data[1:, 6]), label="$c = 1$")
    ax2.semilogx(f_data, np.angle(data[1:, 6]), label="$c = 1$")
    ax1.semilogx(f, np.real(chi_2), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    ax2.semilogx(f, -np.angle(chi_2), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    ax1.legend(loc="upper left", frameon=False)

    # export plot
    print("saving plot")
    plt.subplots_adjust(hspace=0.0)
    #fig.savefig("img/suscept_lif_lin_novikov.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
