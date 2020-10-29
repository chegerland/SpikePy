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
    Delta = 5e-2
    tau_a = 10.0
    lifac = neurons.LIFAC(mu, D, Delta, tau_a)
    lif = lifac.lif

    # calculate susceptibilities of LIF
    f = np.logspace(-3, 1.5, num=300)
    chi_1 = []
    chi_1_adapt = []
    for i in range(len(f)):
        chi_1.append(lif.susceptibility_1(2. * pi * f[i]))
        chi_1_adapt.append(lifac.susceptibility_1(2. * pi * f[i]))

    # prepare plots
    #matplotlib.rcParams["text.usetex"] = True
    #matplotlib.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_ylabel("$ \\Re[\\chi_1(f)] $")
    ax2.set_ylabel("$ \\Im[\\chi_1(f)] $")
    ax2.set_xlabel("$ f $")

    # simulation files to consides
    prefix = "../Spike/data/BigLIFACScan/"
    prefix = "../Spike/data/OldDataCheck/"
    filenames = [
        #            ["../SpikeLibrary/data/LIFAC/BigD/tau_5_Delta_1e-3_suscept.csv", "$\\textrm{numerics}$"],
        #            ["../SpikePlusPlus/data/LIFAC/BigD/tau_5_Delta_1e-3_suscept.csv", "$\\textrm{numerics}$"],
        #["BigD/tau_5_Delta_1e-2_suscept.csv", "$\\mathrm{numerics}$"],
        ["tau_10_Delta_5e-2_Ne4_suscept.csv", "$\\mathrm{numerics}$"],
        ["tau_10_Delta_5e-2_Ne5_suscept.csv", "$\\mathrm{numerics}$"],
        #["tau_10_Delta_5e-2_Ne7_suscept.csv", "$\\mathrm{numerics}$"],
        # ["SmallD/tau_10_Delta1e-3_suscept.csv", "$\\textrm{numerics}$"],
        # ["SmallD/tau_50_Delta3e-2_suscept.csv", "$\\textrm{numerics}$"],
        # ["SmallD/tau_10_Delta3e-2.csv", "$\\textrm{numerics}$"],
    ]

    # plot for every file its contents
    for file in filenames:
        data = np.genfromtxt(prefix + file[0], delimiter=',')
        #data = np.genfromtxt(file[0], delimiter=',')
        f_sim = data[1:, 0]
        chi_real_lin = data[1:, 1]
        chi_imag_lin = data[1:, 2]
        chi_sim_lin = chi_real_lin + 1j * chi_imag_lin
        ax1.semilogx(f_sim, np.real(chi_sim_lin), label=file[1])
        ax2.semilogx(f_sim, -np.imag(chi_sim_lin), label=file[1])

    # plot the theory lines
    ax1.semilogx(f, np.real(chi_1), label="$\chi_1^{\\mathrm{LIF}}(f;\hat{\mu},D)$", color='k', linestyle='--')
    ax2.semilogx(f, np.imag(chi_1), label="$\chi_1^{\\mathrm{LIF}}(f;\hat{\mu},D)$", color='k', linestyle='--')
    ax1.semilogx(f, np.real(chi_1_adapt), label="$\chi_1^{\\mathrm{LIFAC}}(f;\mu,D)$", color='k')
    ax2.semilogx(f, np.imag(chi_1_adapt), label="$\chi_1^{\\mathrm{LIFAC}}(f;\mu,D)$", color='k')
    ax1.legend(loc="lower left", frameon=False)
    ax2.legend(loc="lower right", frameon=False)

    # export plot
    #fig.savefig("img/suscept_lifac_lin.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
