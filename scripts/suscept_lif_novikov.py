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
    #data_path = "../Spike/data/LIF_Novikov/lif_suscept_novikov_N1e7.csv"
    #data_path = "../Spike/data/AntiDiag/lif_T_1000_suscept_novikov.csv"
    #data_path = "../Spike/data/LIF_Novikov/lif_suscept_novikov.csv"
    data = np.genfromtxt(data_path, delimiter=',', dtype=np.complex128)
    f_data = data[1:, 0]

    print("calculating susceptibility")
    f_min = -3
    f_max = 1
    steps = 300
    f = np.logspace(f_min, f_max, num=steps)
    chi_1, chi_2_diag, chi_2_antidiag = calculate_novikov_suscept(
        "cache/lif_1.csv", "cache/lif_2_diag.csv", "cache/lif_2_antidiag.csv", f_min, f_max, steps, lif.susceptibility_1, lif.susceptibility_2)

    # prepare plots
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 24
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    ax1.set_xlabel("$ f $")
    ax2.set_xlabel("$ f $")

    # plot the theory lines
    #ax1.set_ylabel("$ |\\chi_1(f)| $")
    #ax2.set_ylabel("$ \\phi[\\chi_1(f)] $")
    #ax1.semilogx(f_data, np.real(data[1:, 1]), label="$c = 0.1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 1]), label="$c = 0.1$")
    #ax1.semilogx(f_data, np.real(data[1:, 4]), label="$c = 0.5$")
    #ax2.semilogx(f_data, np.angle(data[1:, 4]), label="$c = 0.5$")
    #ax1.semilogx(f_data, np.real(data[1:, 7]), label="$c = 1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 7]), label="$c = 1$")
    # ax1.semilogx(f, np.real(
    #    chi_1), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    # ax2.semilogx(f, -np.angle(chi_1),
    #             label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    #ax1.legend(loc="upper left", frameon=False)

    #ax1.set_ylabel("$ |\\chi_2(f,f)| $")
    #ax2.set_ylabel("$ \\phi[\\chi_2(f,f)] $")
    #ax1.semilogx(f_data, 2*np.real(data[1:, 2]), label="$c = 0.1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 2]), label="$c = 0.1$")
    #ax1.semilogx(f_data, 2*np.real(data[1:, 5]), label="$c = 0.5$")
    #ax2.semilogx(f_data, np.angle(data[1:, 5]), label="$c = 0.5$")
    #ax1.semilogx(f_data, 2*np.real(data[1:, 8]), label="$c = 1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 8]), label="$c = 1$")
    # ax1.semilogx(f, np.real(
    #    chi_2_diag), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    # ax2.semilogx(f, -np.angle(chi_2_diag),
    #             label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    #ax1.legend(loc="upper left", frameon=False)

    ax1.set_ylabel("$ |\\chi_2(f,-f)| $")
    ax2.set_ylabel("$ \\phi[\\chi_2(f,-f)] $")
    #ax1.semilogx(f_data, 2*np.real(data[1:, 3]), label="$c = 0.1$")
    #ax2.semilogx(f_data, np.angle(data[1:, 3]), label="$c = 0.1$")
    #ax1.semilogx(f_data, 2*np.real(data[1:, 6]), label="$c = 0.5$")
    #ax2.semilogx(f_data, np.angle(data[1:, 6]), label="$c = 0.5$")
    ax1.semilogx(f_data, 2*np.real(data[1:, 9]), label="$c = 1$")
    ax2.semilogx(f_data, np.angle(data[1:, 9]), label="$c = 1$")
    # ax1.semilogx(f, np.real(
    #    chi_2_antidiag), label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    # ax2.semilogx(f, -np.angle(chi_2_antidiag),
    #             label="$\\mathrm{analytic~solution}$", color='k', linestyle='-')
    ax1.legend(loc="upper left", frameon=False)

    # export plot
    print("saving plot")
    plt.subplots_adjust(hspace=0.0)
    #fig.savefig("img/suscept_lif_lin_novikov.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
