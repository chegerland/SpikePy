#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons
import csv
import os
import logging

from math import pi


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def main():

    # set logging level
    logging.basicConfig(level=logging.INFO)

    # files
    path = "../Spike/data/Firing_rate/"
    ini = "lifac_two_sigs_slow.ini"
    ini_path = path + ini
    pre, ext = os.path.splitext(ini)
    raster_path = path + pre + "_raster.csv"
    rate_path = path + pre + "_firing_rate.csv"

    # read raster plot data
    logging.info("Reading raster plot data.")
    datafile = open(raster_path, 'r')
    datareader = csv.reader(datafile)
    data_raster = []
    for row in datareader:
        # I split the input string based on the comma separator, and cast every elements into a float
        data_raster.append([float(elem) for elem in row])

    # read firing rate data
    logging.info("Reading firing rate data.")
    data = np.genfromtxt(rate_path, delimiter=',')
    t = data[1:, 0]
    rate = data[1:, 1]

    rate = running_mean(rate, 10)
    t = t[0:rate.size]

    # define neurons and signal
    logging.info("Define analytic LIFAC neurons and calculate analytic rates.")
    lifac = neurons.LIFAC.from_ini(ini_path)
    signal = neurons.TwoCosineSignal.from_ini(ini_path)
    rate_ana_lif = signal.firing_rate_nonlinear(lifac.lif, t)
    rate_ana_lifac = signal.firing_rate_nonlinear(lifac, t)
    rate_ana_lifac_lin = signal.firing_rate_linear(lifac.lif, t)

    # plot
    logging.info("Produce the plot.")

    # set time limits
    x_min = 20
    x_max = 400

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax1.set_ylabel("$r(t)$")
    ax1.set_xlim(x_min, x_max)
    #ax1.set_ylim(0, 0.9)
    ax1.plot(t, rate)
    ax1.plot(t, rate_ana_lif,
             label="$\\mathrm{nonlinear~theory~} \\chi_2^{\\mathrm{LIF}}$")
    ax1.plot(t, rate_ana_lifac,
             label="$\\mathrm{nonlinear~theory~} \\chi_2^{\\mathrm{LIFAC}}$")
    ax1.plot(t, rate_ana_lifac_lin,
             label="$\\mathrm{linear~theory~} \\chi_2^{\\mathrm{LIFAC}}$")
    ax1.legend(loc="upper right", frameon="tight")

    ax2.eventplot(data_raster, linestyles="dotted")
    ax2.set_ylabel("$\mathrm{trials}$")
    ax2.set_xlabel("$t$")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, 100)

    fig.savefig("img/"+pre+".png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
