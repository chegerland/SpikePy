#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from spike import LIF
from utils import calculate_novikov_suscept

from math import pi


def main():
    # define LIFAC and LIF
    mu = 1.1
    D = 1e-3
    lif = LIF(mu, D)

    print("loading data")
    # read data
    data = np.genfromtxt("../Spike/data/LIF_Novikov/lif_suscept_novikov_N1e7.csv", delimiter=',', dtype=np.complex128)
    f_data = data[1:, 0]

    print("calculating susceptibility")
    f_min = -3
    f_max = 1.5
    steps = 300
    f = np.logspace(f_min, f_max, num=steps)
    chi_1, chi_2 = calculate_novikov_suscept("cache/lif_1.ini", "cache/lif_2.ini", f_min, f_max, steps, lif.susceptibility_1, lif.susceptibility_2)

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
    #ax1.semilogx(f_data, np.real(data[1:, 1]), label="$c = 0.01$")
    #ax2.semilogx(f_data, np.angle(data[1:, 1]), label="$c = 0.01$")
    #ax1.semilogx(f_data, np.real(data[1:, 3]), label="$c = 0.1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 3]), label="$c = 0.1$")
    #ax1.semilogx(f_data, np.real(data[1:, 5]), label="$c = 1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 5]), label="$c = 1$")
    #ax1.semilogx(f, np.real(chi_1), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    #ax2.semilogx(f, -np.angle(chi_1), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    #ax1.legend(loc="upper left", frameon=False)

    #ax1.semilogx(f_data, 2*np.real(data[1:, 2]), label="$c = 0.01$")
    #ax2.semilogx(f_data, np.angle(data[1:, 2]), label="$c = 0.01$")
    ax1.semilogx(f_data, 2*np.real(data[1:, 4]), label="$c = 0.1$")
    ax2.semilogx(f_data, np.angle(data[1:, 4]), label="$c = 0.1$")
    ax1.semilogx(f_data, 2*np.real(data[1:, 6]), label="$c = 1$")
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
