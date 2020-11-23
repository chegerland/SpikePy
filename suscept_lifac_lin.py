#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons

from math import pi


def main():
    # define LIFAC and LIF
    params = [[2.0, 1.0, 0.01, 5.0], [2.0, 0.1, 0.1, 20.0], [2.0, 1.0, 0.01, 0.5], [2.0, 0.1, 0.5, 1.0]]
    lifacs = []
    for i in range(4):
        lifacs.append(neurons.LIFAC(params[i][0], params[i][1], params[i][2], params[i][3]))

    # calculate susceptibilities of LIF and LIFAC
    f = np.logspace(-3, 2, num=300)
    chi_1 = np.zeros(shape=(4,len(f)), dtype=np.complex128)
    chi_1_adapt = np.zeros(shape=(4,len(f)), dtype=np.complex128)
    for i in range(len(f)):
        for j in range(4):
            chi_1[j][i] = lifacs[j].lif.susceptibility_1(2. * pi * f[i])
            chi_1_adapt[j][i] = lifacs[j].susceptibility_1(2. * pi * f[i])

    # prepare plots
    #matplotlib.rcParams["text.usetex"] = True
    #matplotlib.rcParams["font.size"] = 20
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    #ax1.set_ylabel("$ \\Re[\\chi_1(f)] $")
    #ax2.set_ylabel("$ \\Im[\\chi_1(f)] $")
    #ax2.set_xlabel("$ f $")

    # read data
    prefix = "../Spike/data/LIFAC_Lin_Scan/"
    filenames = [
        ["slow_Delta_small_suscept.csv", "$\\mathrm{slow~small}$"],
        ["slow_Delta_big_suscept.csv", "$\\mathrm{slow~big}$"],
        ["fast_Delta_small_suscept.csv", "$\\mathrm{fast~small}$"],
        ["fast_Delta_big_suscept.csv", "$\\mathrm{fast~big}$"],
    ]

    # plot for every file its contents
    for i in range(4):
        data = np.genfromtxt(prefix + filenames[i][0], delimiter=',')
        f_sim = data[1:, 0]
        chi_real = data[1:, 1]
        chi_imag = data[1:, 2]
        chi_data = chi_real + 1j * chi_imag

        # plot data
        ax[0][i].semilogx(f_sim, np.abs(chi_data), label=filenames[i][1])
        ax[1][i].semilogx(f_sim, -np.angle(chi_data), label=filenames[i][1])

        # plot theory
        ax[0][i].semilogx(f, np.abs(chi_1[i]), label="$\mathrm{LIF}$", linestyle='--', color='r')
        ax[1][i].semilogx(f, np.angle(chi_1[i]), label="$\mathrm{LIF}$", linestyle='--', color='r')
        ax[0][i].semilogx(f, np.abs(chi_1_adapt[i]), label="$\mathrm{LIFAC}$", linestyle='--', color='k')
        ax[1][i].semilogx(f, np.angle(chi_1_adapt[i]), label="$\mathrm{LIFAC}$", linestyle='--', color='k')

        # legend
        ax[0][i].legend(loc="lower left", frameon=False)
        ax[1][i].legend(loc="lower left", frameon=False)

    # export plot
    #fig.savefig("img/suscept_lifac_lin_compar_better_better.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
