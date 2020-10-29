#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons

from math import pi


def main():

    # read data
    file = "../Spike/data/FiringRate/lif_firing_rate.csv"
    data = np.genfromtxt(file, delimiter=',')
    t = data[1:,0]
    rate = data[1:,1]

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$r(t)$")

    ax.plot(t, rate)

    plt.show()


if __name__ == "__main__":
    main()
