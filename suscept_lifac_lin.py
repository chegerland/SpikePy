#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons
import configparser
import os

from math import pi


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def lifac_from_config(path_to_ini):
    config = configparser.ConfigParser()
    config.read(path_to_ini)

    params = config['Neuron']
    assert (params['type'] == "LIFAC")

    lifac = neurons.LIFAC(float(params['mu']), float(params['D']), float(params['Delta']), float(params['tau_a']))
    return lifac


def calculate_suscept_lin(lifac):
    f = np.logspace(-3, 2, num=300)
    chi_1 = np.zeros(shape=(len(f)), dtype=complex)
    chi_1_lif = np.zeros(shape=(len(f)), dtype=complex)
    for i in range(len(f)):
        chi_1[i] = lifac.susceptibility_1(2. * pi * f[i])
        chi_1_lif[i] = lifac.lif.susceptibility_1(2. * pi * f[i])

    return f, chi_1, chi_1_lif


def plot_suscept_lin(ini):
    # ini_path = "../Spike/data/LIFAC_Lin/lifac_tau_20_Delta_1e-1.ini"
    path = "../Spike/data/LIFAC_Lin/"
    ini_path = path + ini
    pre, ext = os.path.splitext(ini)
    csv_path = path + pre + "_suscept_lin.csv"
    # print(pre, ext, ini_path, csv_path)

    # define LIFAC and LIF
    lifac = lifac_from_config(ini_path)

    # calculate susceptibilities of LIF
    f, chi_1, chi_1_lif = calculate_suscept_lin(lifac)

    # prepare plots
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 42
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20), sharex=True)

    ax1.set_ylabel("$ |\\chi_1(f)| $")
    ax2.set_ylabel("$ \\phi[\\chi_1(f)] $")
    ax2.set_xlabel("$ f $")

    # read data
    data = np.genfromtxt(csv_path, delimiter=',', dtype=complex)
    chi_1_num = data[1:, 1]
    f_num = data[1:, 0]

    # plot the theory lines
    ax1.semilogx(f_num, np.abs(chi_1_num), label="$\\mathrm{data}$", linewidth=4.0)
    ax2.semilogx(f_num, np.angle(chi_1_num), label="$\\mathrm{data}$", linewidth=4.0)
    ax1.semilogx(f, np.abs(chi_1), label="$\\chi_1^{\\mathrm{LIFAC}}(f;\\mu,D)$", color='k', linestyle='-',
                 linewidth=4.0)
    ax2.semilogx(f, -np.angle(chi_1), label="$\\chi_1^{\\mathrm{LIFAC}}(f;\\mu,D)$", color='k', linestyle='-',
                 linewidth=4.0)
    ax1.semilogx(f, np.abs(chi_1_lif), label="$\\chi_1^{\\mathrm{LIF}}(f;\\hat{\\mu},D)$", color='k', linestyle='--',
                 linewidth=4.0)
    ax2.semilogx(f, -np.angle(chi_1_lif), label="$\\chi_1^{\\mathrm{LIF}}(f;\\mu,D)$", color='k', linestyle='--',
                 linewidth=4.0)
    ax1.legend(loc="upper right", frameon=False)
    # ax2.legend(loc="lower left", frameon=False)

    ax1.set_xlim(0.001, 100)
    ax2.set_xlim(0.001, 100)

    plt.subplots_adjust(hspace=0.0)

    # export plot
    fig.savefig("img/" + pre + ".png", bbox_inches="tight")

    plt.show()


def main():
    inis = ["lifac_A.ini",
            "lifac_B.ini",
            "lifac_C.ini",
            "lifac_D.ini",
            "lifac_E.ini",
            ]

    for ini in inis:
        plot_suscept_lin(ini)
    # plot_suscept_lin("lifac_F.ini")


if __name__ == "__main__":
    main()
