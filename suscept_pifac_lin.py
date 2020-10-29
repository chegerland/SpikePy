#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons

from math import pi


def main():
    # define LIFAC and LIF
    mu = 2.5
    D = 1.0
    tau_a = 10.0*mu
    Delta = 0.12
    pifac = neurons.PIFAC(mu, D, Delta, tau_a)
    pif = pifac.pif

    # calculate susceptibilities of LIF
    f = np.logspace(-3, 1.5, num=300)
    chi_1 = []
    chi_1_adapt = []
    for i in range(len(f)):
        chi_1.append(pif.susceptibility_1(2. * pi * f[i]))
        chi_1_adapt.append(pifac.susceptibility_1(2. * pi * f[i]))

    # prepare plots
    #matplotlib.rcParams["text.usetex"] = True
    #matplotlib.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.set_ylabel("$ |\\chi_1(f)| $")
    ax2.set_ylabel("$ \\mathrm{arg}[\\chi_1(f)] $")
    ax2.set_xlabel("$ f $")

    # simulation files to consides
    prefix = "../Spike/data/PIFAC/"
    filenames = [
        ["schwalger_pif_A_2_suscept.csv", "$\\mathrm{numerics}$"],
        #["schwalger_pif_A_suscept.csv", "$\\mathrm{numerics}$"],
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
    ax1.semilogx(f, np.real(chi_1), label="$\chi_1^{\\mathrm{PIF}}(f;\hat{\mu},D)$", color='k', linestyle='--')
    ax2.semilogx(f, np.imag(chi_1), label="$\chi_1^{\\mathrm{PIF}}(f;\hat{\mu},D)$", color='k', linestyle='--')
    ax1.semilogx(f, np.real(chi_1_adapt), label="$\chi_1^{\\mathrm{PIFAC}}(f;\mu,D)$", color='k')
    ax2.semilogx(f, np.imag(chi_1_adapt), label="$\chi_1^{\\mathrm{PIFAC}}(f;\mu,D)$", color='k')
    ax1.legend(loc="lower left", frameon=False)
    ax2.legend(loc="lower right", frameon=False)

    # export plot
    #fig.savefig("img/suscept_pifac_lin.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
