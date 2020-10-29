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

    # calculate susceptibilities of LIF
    f = np.logspace(-3, 1.5, num=300)
    chi_2_adapt = []
    for i in range(len(f)):
        chi_2_adapt.append(lifac.susceptibility_2(2.*pi*f[i], 2.*pi*f[i]))

    # prepare plots
    #matplotlib.rcParams["text.usetex"] = True
    #matplotlib.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,15), sharex=True)
    ax1.set_ylabel("$ \\Re[\\chi_2(f, f)] $")
    ax2.set_ylabel("$ \\Im[\\chi_2(f, f)] $")
    ax1.set_xlabel("$ f $")

    # simulation files to consides
    prefix = "../Spike/data/BigLIFACScan/"
    filenames = [
        ["BigD/tau_5_Delta_1e-2_suscept.csv", "$\\mathrm{numerics}$"],
    ]

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    # plot for every file its contents
    for file in filenames:
        data = np.genfromtxt(prefix + file[0], delimiter=',')
        f_sim = data[1:, 0]
        chi_real_nonlin = data[1:, 3]
        chi_imag_nonlin = data[1:, 4]

        chi_real_nonlin_moving = running_mean(chi_real_nonlin, 50)
        chi_imag_nonlin_moving = running_mean(chi_imag_nonlin, 50)
        f_sim_moving = f_sim[0:chi_real_nonlin_moving.size]

        ax1.semilogx(f_sim, 2*chi_real_nonlin, label=file[1])
        ax2.semilogx(f_sim, -2*chi_imag_nonlin, label=file[1])
        ax1.semilogx(f_sim_moving, 2*chi_real_nonlin_moving, label=file[1]+" + moving average")
        ax2.semilogx(f_sim_moving, -2*chi_imag_nonlin_moving, label=file[1] + " + moving average")


    # plot the theory lines
    ax1.semilogx(f, np.real(chi_2_adapt), label="$\chi_2^{\\mathrm{LIFAC}}(f, f; \mu, D)$", color='k')
    ax2.semilogx(f, np.imag(chi_2_adapt), label="$\chi_2^{\\mathrm{LIFAC}}(f, f; \mu, D)$", color='k')
    ax1.legend(loc="lower left", frameon=False)
    ax2.legend(loc="lower left", frameon=False)

    # export plot
    fig.savefig("img/suscept_nonlin_lifac_moving_average.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
