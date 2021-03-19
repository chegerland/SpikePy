#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons
import utils


def main():
    """
    We plot the second order susceptibility of the LIF next to numerically obtained data from Spike.
    """
    # define the lif neuron
    lif = neurons.LIF(1.1, 0.001)
    chi_2_numeric = utils.calculate_analytic_matrix("lif_matrix.csv", 0.0, 0.8, 79, lif.susceptibility_2)
    chi_2 = utils.create_full_matrix(chi_2_numeric)

    # read numeric matrix from file
    data = np.genfromtxt("../Spike/data/LIF_Matrix/lif_suscept_matrix.csv", delimiter=',',
                         dtype=np.complex128)
    size_data = data.shape[0]
    for i in range(size_data):
        for j in range(size_data):
            if i <= j:
                data[i][j] = np.conj(data[i][j])

    max_freq_bin = 80
    data = data[1:max_freq_bin, 1:max_freq_bin]
    chi_2_data = utils.create_full_matrix(data)

    # plotting
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 20
    matplotlib.rcParams['axes.titlepad'] = 20
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    kwargs_abs = {
        'extent': [-max_freq_bin / 100, max_freq_bin / 100, -max_freq_bin / 100, max_freq_bin / 100],
        'vmax': 430,
    }
    kwargs_angle = {
        'extent': [-max_freq_bin / 100, max_freq_bin / 100, -max_freq_bin / 100, max_freq_bin / 100],
    }

    ax1.set_title("$|\\chi_2^{\\mathrm{LIF(num)}}(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax1, np.abs(chi_2_data), **kwargs_abs)

    ax2.set_title("$|\\chi_2^{\\mathrm{LIF (ana)}}(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax2, np.abs(chi_2), **kwargs_abs)

    ax3.set_title("$\\phi(\\chi_2^{\\mathrm{LIF(num)}}(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax3, np.angle(chi_2_data), **kwargs_angle)

    ax4.set_title("$\\phi(\\chi_2^{\\mathrm{LIF(ana)}}(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax4, np.angle(chi_2), **kwargs_angle)

    # save the file and show the plot
    fig.savefig("img/suscept_lif_matrix.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
