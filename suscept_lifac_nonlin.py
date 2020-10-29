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
    Delta = 1e-2
    tau_a = 5.0
    lifac = neurons.LIFAC(mu, D, Delta, tau_a)
    lif = lifac.lif

    # calculate susceptibilities of LIF
    f = np.logspace(-3, 1.5, num=300)
    chi_1 = []
    chi_2 = []
    chi_1_adapt = []
    chi_2_adapt = []
    for i in range(len(f)):
        chi_1.append(lif.susceptibility_1(2.*pi*f[i]))
        chi_1_adapt.append(lifac.susceptibility_1(2.*pi*f[i]))
        chi_2.append(lif.susceptibility_2(2.*pi*f[i], 2.*pi*f[i]))
        chi_2_adapt.append(lifac.susceptibility_2(2.*pi*f[i], 2.*pi*f[i]))

    # prepare plots
    #matplotlib.rcParams["text.usetex"] = True
    #matplotlib.rcParams["font.size"] = 20
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15), sharex=True)
    ax1.set_ylabel("$ \\Re[\\chi_1(f)] $")
    ax2.set_ylabel("$ \\Im[\\chi_1(f)] $")
    ax2.set_xlabel("$ f $")
    ax3.set_ylabel("$ \\Re[\\chi_2(f, f)] $")
    ax4.set_ylabel("$ \\Im[\\chi_2(f, f)] $")
    ax3.set_xlabel("$ f $")

    # simulation files to consides
    prefix = "../Spike/data/BigLIFACScan/"
    #prefix = "../Spike/data/OldDataCheck/"
    filenames = [
        #            ["../SpikeLibrary/data/LIFAC/BigD/tau_5_Delta_1e-3_suscept.csv", "$\\textrm{numerics}$"],
        #            ["../SpikePlusPlus/data/LIFAC/BigD/tau_5_Delta_1e-3_suscept.csv", "$\\textrm{numerics}$"],
        ["BigD/tau_5_Delta_1e-2_suscept.csv", "$\\mathrm{numerics}$"],
        #["tau_10_Delta_5e-2_Ne4_suscept.csv", "$\\mathrm{numerics}$"],
        #["tau_10_Delta_5e-2_Ne5_suscept.csv", "$\\mathrm{numerics}$"],
        # ["SmallD/tau_10_Delta1e-3_suscept.csv", "$\\textrm{numerics}$"],
        #["SmallD/tau_50_Delta_1e-2_suscept.csv", "$\\mathrm{numerics}$"],
        # ["SmallD/tau_10_Delta3e-2.csv", "$\\textrm{numerics}$"],
    ]

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    # plot for every file its contents
    for file in filenames:
        data = np.genfromtxt(prefix + file[0], delimiter=',')
        f_sim = data[1:, 0]
        chi_real_lin = data[1:, 1]
        chi_imag_lin = data[1:, 2]
        chi_sim_lin = chi_real_lin +1j*chi_imag_lin
        chi_real_nonlin = data[1:, 3]
        chi_imag_nonlin = data[1:, 4]

        chi_real_nonlin = running_mean(chi_real_nonlin, 50)
        chi_imag_nonlin = running_mean(chi_imag_nonlin, 50)
        f_sim_2 = f_sim[0:chi_real_nonlin.size]

        chi_sim_nonlin = chi_real_nonlin +1j*chi_imag_nonlin
        ax1.semilogx(f_sim, np.real(chi_sim_lin), label=file[1])
        ax2.semilogx(f_sim, -np.imag(chi_sim_lin), label=file[1])
        ax3.semilogx(f_sim_2, 2*np.real(chi_sim_nonlin), label=file[1])
        ax4.semilogx(f_sim_2, -2*np.imag(chi_sim_nonlin), label=file[1])



    # plot the theory lines
    ax1.semilogx(f, np.real(chi_1), label="$\chi_1^{\\mathrm{LIF}}(f; \hat{\mu}, D)$", color='k', linestyle='--')
    ax2.semilogx(f, np.imag(chi_1), label="$\chi_1^{\\mathrm{LIF}}(f; \hat{\mu}, D)$", color='k', linestyle='--')
    ax3.semilogx(f, np.real(chi_2), label="$\chi_2^{\\mathrm{LIF}}(f, f; \hat{\mu}, D)$", color='k', linestyle='--')
    ax4.semilogx(f, np.imag(chi_2), label="$\chi_2^{\\mathrm{LIF}}(f, f; \hat{\mu}, D)$", color='k', linestyle='--')
    ax1.semilogx(f, np.real(chi_1_adapt), label="$\chi_1^{\\mathrm{LIF}}(f; \mu, D)$", color='k')
    ax2.semilogx(f, np.imag(chi_1_adapt), label="$\chi_1^{\\mathrm{LIF}}(f; \mu, D)$", color='k')
    ax3.semilogx(f, np.real(chi_2_adapt), label="$\chi_2^{\\mathrm{LIFAC}}(f, f; \mu, D)$", color='k')
    ax4.semilogx(f, np.imag(chi_2_adapt), label="$\chi_2^{\\mathrm{LIFAC}}(f, f; \mu, D)$", color='k')
    ax1.legend(loc="lower left", frameon=False)
    ax2.legend(loc="lower left", frameon=False)
    ax3.legend(loc="upper left", frameon=False)
    ax4.legend(loc="upper left", frameon=False)

    # export plot
    #fig.savefig("suscept_lifac_lin_nonlin_2.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
