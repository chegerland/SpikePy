#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import neurons
import utils
import matplotlib


def main():
    """
    We plot a comparison between the second order susceptibility for the LIFAC and the associated LIF.
    """

    # define lifac neuron
    lifac = neurons.LIFAC(3.5, 0.1, 0.1, 10.0)

    # calculate analytic matrices and construct the full matrix from them
    chi_2_numeric_lifac = utils.calculate_analytic_matrix("chi_2_lifac_1.csv", 0.0, 3.5, 100,
                                                          lifac.susceptibility_2_better)
    chi_2_lifac = utils.create_full_matrix(chi_2_numeric_lifac)
    chi_2_numeric_lif = utils.calculate_analytic_matrix("chi_2_lifac_lif.csv", 0.0, 3.5, 100,
                                                        lifac.lif.susceptibility_2)
    chi_2_lif = utils.create_full_matrix(chi_2_numeric_lif)

    max_freq_bin = 350

    # plotting
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.size"] = 20
    matplotlib.rcParams['axes.titlepad'] = 20
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 30))

    kwargs_abs = {
        'extent': [-max_freq_bin / 100, max_freq_bin / 100, -max_freq_bin / 100, max_freq_bin / 100],
        'vmin': 0,
        'vmax': 0.5,
    }
    kwargs_angle = {
        'extent': [-max_freq_bin / 100, max_freq_bin / 100, -max_freq_bin / 100, max_freq_bin / 100],
    }

    ax1.set_title("$|\\chi_2^{\\mathrm{LIFAC}}(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax1, np.abs(chi_2_lifac), **kwargs_abs)

    ax2.set_title("$|\\chi_2^{\\mathrm{LIF}}(f_1, f_2)|$")
    utils.plot_absolute_value(fig, ax2, np.abs(chi_2_lif), **kwargs_abs)

    ax3.set_title("$\\phi(\\chi_2^{\\mathrm{LIFAC}}(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax3, np.angle(chi_2_lifac), **kwargs_angle)

    ax4.set_title("$\\phi(\\chi_2^{\\mathrm{LIF}}(f_1, f_2))$")
    utils.plot_complex_angle(fig, ax4, np.angle(chi_2_lif), **kwargs_angle)

    # save the file and show the plot
    fig.savefig("img/suscept_lifac_matrix_lif.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
