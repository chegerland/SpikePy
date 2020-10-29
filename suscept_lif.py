#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons

from math import pi


def main():
    # define LIFAC and LIF
    mu = 3.5
    D = 1e-1
    lif = neurons.LIF(mu, D)

    # calculate susceptibilities of LIF
    f = np.logspace(-7, 1.5, num=500)
    chi_1 = []
    chi_2 = []
    chi_2_2 = []
    for i in range(len(f)):
        chi_1.append(lif.susceptibility_1(2. * pi * f[i]))
        chi_2.append(lif.susceptibility_2(0, 2. * pi * f[i]))
        chi_2_2.append(lif.susceptibility_2(1e-7, 2. * pi * f[i]))

    # prepare plots
    #matplotlib.rcParams["text.usetex"] = True
    #matplotlib.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_ylabel("$ \\Re[\\chi_1(f)] $")
    ax2.set_ylabel("$ \\Im[\\chi_1(f)] $")
    ax2.set_xlabel("$ f $")

    # plot the theory lines
    ax1.semilogx(f, np.abs(chi_1), label="$\chi_1^{\\mathrm{LIF}}(f;\hat{\mu},D)$", color='k', linestyle='-')
    ax2.semilogx(f, np.angle(chi_1), label="$\chi_1^{\\mathrm{LIF}}(f;\hat{\mu},D)$", color='k', linestyle='-')
    ax1.semilogx(f, np.abs(chi_2), label="$\chi_2^{\\mathrm{LIF}}(f;\hat{\mu},D)$", color='b', linestyle='-')
    ax2.semilogx(f, np.angle(chi_2), label="$\chi_2^{\\mathrm{LIF}}(f;\hat{\mu},D)$", color='b', linestyle='-')
    ax1.semilogx(f, np.abs(chi_2_2), label="$\chi_2^{\\mathrm{LIF}}(f;\hat{\mu},D)$", color='r', linestyle='-')
    ax2.semilogx(f, np.angle(chi_2_2), label="$\chi_2^{\\mathrm{LIF}}(f;\hat{\mu},D)$", color='r', linestyle='-')
    ax1.legend(loc="lower left", frameon=False)
    ax2.legend(loc="lower right", frameon=False)

    # export plot
    #fig.savefig("img/suscept_lifac_lin.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
