#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons


def delta_fixed():
    # prepare plot
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 24
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("$r_0$")

    # calculate analytic data
    mu_array = np.linspace(0.0, 5.0, num=100)
    rate = np.zeros(shape=100)
    # tau_list = [1e-2, 1e-1, 1.0, 5.0, 10.0]
    tau_list = [1e-1, 1.0, 5.0, 10.0]

    for tau_a in tau_list:
        for i in range(mu_array.size):
            lifac = neurons.LIFAC(mu_array[i], 0.1, 1e-1, tau_a)
            rate[i] = lifac.stationary_rate()

        ax.plot(mu_array, rate, color="black")

    # read data
    file = "../Spike/data/r0_mu_tau_a_lifac.csv"
    data = np.genfromtxt(file, delimiter=',')
    ax.set_title("$\\Delta = 0.1$")
    ax.set_xlim(0, 5)

    # plot data
    # ax.plot(data[0:, 0], data[0:, 1], "o", label="$\tau_a = 0.01$")
    ax.plot(data[0:, 0], data[0:, 2], "o", label="$\\tau_a = 0.1$")
    ax.plot(data[0:, 0], data[0:, 3], "o", label="$\\tau_a = 1$")
    ax.plot(data[0:, 0], data[0:, 4], "o", label="$\\tau_a = 5$")
    ax.plot(data[0:, 0], data[0:, 5], "o", label="$\\tau_a = 10$")

    ax.legend(loc="upper left", frameon=False)

    plt.show()

    fig.savefig("img/r0_mu_Delta_fixed.pgf", bbox_inches="tight")


def tau_a_fixed():
    # prepare plot
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 24
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("$r_0$")

    # calculate analytic data
    mu_array = np.linspace(0.0, 5.0, num=100)
    rate = np.zeros(shape=100)
    # Delta_list = [1e-2, 1e-1, 1.0, 5.0, 10.0]
    delta_list = [1e-2, 1e-1, 1.0, 10.0]

    for Delta in delta_list:
        for i in range(mu_array.size):
            lifac = neurons.LIFAC(mu_array[i], 0.1, Delta, 5.0)
            rate[i] = lifac.stationary_rate()

        ax.plot(mu_array, rate, color="black")

    # read data
    file = "../Spike/data/r0_mu_Delta_lifac.csv"
    data = np.genfromtxt(file, delimiter=',')
    ax.set_title("$\\tau_a = 5$")
    ax.set_xlim(0, 5)

    # plot data
    ax.plot(data[0:, 0], data[0:, 1], "o", label="$\\Delta = 0.01$")
    ax.plot(data[0:, 0], data[0:, 2], "o", label="$\\Delta = 0.1$")
    ax.plot(data[0:, 0], data[0:, 3], "o", label="$\\Delta = 1$")
    # ax.plot(data[0:, 0], data[0:, 4], "o", label="$\\Delta = 5$")
    ax.plot(data[0:, 0], data[0:, 5], "o", label="$\\Delta = 10$")

    ax.legend(loc="upper left", frameon=False)

    plt.show()

    fig.savefig("img/r0_mu_tau_fixed.pgf", bbox_inches="tight")


def main():
    delta_fixed()
    tau_a_fixed()


if __name__ == "__main__":
    main()
