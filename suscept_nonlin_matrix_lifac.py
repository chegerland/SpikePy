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
    lifac = neurons.LIFAC(3.5, 1e-1, 1e-1, 10.0)

    # calculate the analytic matrices for the full range
    chi_2_numeric_0 = utils.calculate_analytic_matrix("chi_2_lifac_0.csv", 0.0, 3.5, 100, lifac.susceptibility_2)
    chi_2_0 = utils.create_full_matrix(chi_2_numeric_0)
    chi_2_numeric_1 = utils.calculate_analytic_matrix("chi_2_lifac_1.csv", 0.0, 3.5, 100, lifac.susceptibility_2_better)
    chi_2_1 = utils.create_full_matrix(chi_2_numeric_1)

    # import the data
    data = np.genfromtxt("../Spike/data/LIFAC_Matrix/lifac_Ne6_suscept_matrix.csv", delimiter=',',
                         dtype=np.complex128)
    size_data = data.shape[0]
    for i in range(size_data):
        for j in range(size_data):
            if i <= j:
                data[i][j] = np.conj(data[i][j])

    max_freq_bin = 350
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
        'vmin': 0,
        'vmax': 0.5
    }
    kwargs_angle = {
        'extent': [-max_freq_bin / 100, max_freq_bin / 100, -max_freq_bin / 100,
                   max_freq_bin / 100]
    }

    ax1.set_title("$|\\chi_2^{\\mathrm{LIFAC(num)}}(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax1, np.abs(chi_2_data), **kwargs_abs)

    ax2.set_title("$|\\chi_2^0(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax2, np.abs(chi_2_0), **kwargs_abs)

    ax3.set_title("$|\\chi_2^1(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax3, np.abs(chi_2_1), **kwargs_abs)

    ax4.set_title("$\\phi(\\chi_2^{\\mathrm{LIFAC(num)}}(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax4, np.angle(chi_2_data), **kwargs_angle)

    ax5.set_title("$\\phi(\\chi_2^0(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax5, np.angle(chi_2_0), **kwargs_angle)

    ax6.set_title("$\\phi(\\chi_2^1(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax6, np.angle(chi_2_1), **kwargs_angle)

    # save the file and show the plot
    fig.savefig("img/suscept_lifac_matrix_orders.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
