#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def trajectory():
    # prepare plot
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 24
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex="all")
    ax[1].set_xlabel("$t$")
    ax[0].set_ylabel("$a$")
    ax[1].set_ylabel("$v$")

    # read data
    file = "../Spike/data/lifac_trajectory.csv"
    data = np.genfromtxt(file, delimiter=',')

    # trim first bit
    cut_low = 0 * 1000
    cut_high = 3 * 1000
    data = data[cut_low:cut_high, :]

    # plot data
    # ax.plot(data[0:, 0], data[0:, 1], "o", label="$\tau_a = 0.01$")
    ax[0].plot(data[0:, 0], data[0:, 2], color="black")
    ax[1].plot(data[0:, 0], data[0:, 1], color="black")

    ax[0].set_xlim(0, 3)
    ax[1].set_xlim(0, 3)

    plt.show()

    fig.savefig("img/trajectory_lifac.pgf", bbox_inches="tight")


def phase_plot():
    # prepare plot
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 24
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xlabel("$v$")
    ax.set_ylabel("$a$")

    # read data
    file = "../Spike/data/lifac_trajectory_no_noise.csv"
    data = np.genfromtxt(file, delimiter=',')

    # trim first bit
    cut_low = 0 * 1000
    cut_high = 3 * 1000
    data = data[cut_low:cut_high, :]

    # plot data
    ax.plot(data[0:, 1], data[0:, 2], color="black")

    plt.show()

    fig.savefig("img/phase_plot_lifac.pgf", bbox_inches="tight")


def main():
    phase_plot()


if __name__ == "__main__":
    main()
