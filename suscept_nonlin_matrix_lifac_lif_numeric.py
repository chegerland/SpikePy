#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import neurons
import utils


def main():
    """
    We plot the different approximation orders for the second order susceptibility of the LIFAC and also numerically
    obtained data from Spike.
    """

    # define the lif neuron
    lifac = neurons.LIFAC(3.5, 1e-2, 1e-1, 20.0)

    # frequency range
    f_min = 0.0
    f_max = 1.2
    steps = 50

    # calculate analytic matrices and construct the full matrix from them
    chi_2_numeric_lifac = utils.calculate_analytic_matrix("chi_2_lifac.csv", f_min, f_max, steps,
                                                          lifac.susceptibility_2)
    chi_2_lifac = utils.create_full_matrix(chi_2_numeric_lifac)
    chi_2_numeric_lif = utils.calculate_analytic_matrix("chi_2_lifac_lif.csv", f_min, f_max, steps,
                                                        lifac.lif.susceptibility_2)
    chi_2_lif = utils.create_full_matrix(chi_2_numeric_lif)

    # import the data
    data = np.genfromtxt("../Spike/data/LIFAC_LIF_MATRIX_COMPAR/lifac_low_noise_suscept_matrix.csv", delimiter=',',
                         dtype=np.complex128)
    size_data = data.shape[0]
    for i in range(size_data):
        for j in range(size_data):
            if i <= j:
                data[i][j] = np.conj(data[i][j])

    max_freq_bin = 120
    data = data[1:max_freq_bin, 1:max_freq_bin]
    chi_2_data = utils.create_full_matrix(data)

    # plotting
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 20
    matplotlib.rcParams['axes.titlepad'] = 20
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(30, 20))

    kwargs_abs = {
        'extent': [-max_freq_bin / 100, max_freq_bin / 100, -max_freq_bin / 100,
                   max_freq_bin / 100],
        #'vmin': 0,
        #'vmax': 0.5
    }
    kwargs_angle = {
        'extent': [-max_freq_bin / 100, max_freq_bin / 100, -max_freq_bin / 100,
                   max_freq_bin / 100]
    }

    ax1.set_title("$|\\chi_2^{\\mathrm{LIFAC(num)}}(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax1, np.abs(chi_2_data), **kwargs_abs)

    ax2.set_title("$|\\chi_2^{LIFAC}(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax2, np.abs(chi_2_lifac), **kwargs_abs)

    ax3.set_title("$|\\chi_2^{LIF}(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax3, np.abs(chi_2_lif), **kwargs_abs)

    ax4.set_title("$\\phi(\\chi_2^{\\mathrm{LIFAC(num)}}(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax4, np.angle(chi_2_data), **kwargs_angle)

    ax5.set_title("$\\phi(\\chi_2^{LIFAC}(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax5, np.angle(chi_2_lifac), **kwargs_angle)

    ax6.set_title("$\\phi(\\chi_2^{LIF}(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax6, np.angle(chi_2_lif), **kwargs_angle)

    # save the file and show the plot
    fig.savefig("img/suscept_lifac_matrix_orders.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
