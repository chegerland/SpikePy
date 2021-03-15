#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import neurons
import matplotlib
from math import pi


def main():
    """
    We analyze the factor C(omega)
    """

    # define lifac neuron (mu, D, Delta, tau_a)
    lifac = neurons.LIFAC(3.5, 0.1, 1e-1, 20.0)

    # frequency scale
    f = np.logspace(-3, 1.5, num=100)
    c = np.zeros(f.shape[0], dtype=complex)
    for i in range(len(f)):
        c[i] = lifac.c_function(2. * pi * f[i])

    # plotting
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 26
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1.set_ylabel("$|C(f)| $")
    ax2.set_ylabel("$ \\phi[C(f)] $")
    ax2.set_xlabel("$ f $")

    # plot the theory lines
    ax1.semilogx(f, np.abs(c), color='k', linestyle='-')
    ax2.semilogx(f, np.angle(c), color='k', linestyle='-')

    # save the file and show the plot
    plt.show()


if __name__ == "__main__":
    main()
