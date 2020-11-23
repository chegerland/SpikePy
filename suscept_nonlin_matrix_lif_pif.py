#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import neurons

from math import pi


def main():
    # analytics
    #lif = neurons.LIF(1.1, 0.001)
    #steps = 100
    #chi_2 = np.zeros(shape=(steps, steps), dtype=complex)
    #f = np.linspace(0.0, 1.0, num=steps)

    #for i in range(steps):
    #    for j in range(steps):
    #        chi_2[i, j] = lif.susceptibility_2(2.0 * pi * f[i], 2.0 * pi * f[j])

    # data
    # data = np.genfromtxt("../Spike/data/NonlinMatrix/lif_suscept_matrix.csv", delimiter=',')
    # data = np.genfromtxt("../Spike/data/NonlinMatrix/lif_Ne5_suscept_matrix.csv", delimiter=',')
    # data = np.genfromtxt("../Spike/data/NonlinMatrix/lifac_suscept_matrix.csv", delimiter=',')
    data_lif = np.genfromtxt("../Spike/data/LIF_Check/lif_suscept_matrix.csv", delimiter=',', dtype=np.complex128)
    data_pif = np.genfromtxt("../Spike/data/LIF_Check/pif_suscept_matrix.csv", delimiter=',', dtype=np.complex128)

    # data = scipy.ndimage.gaussian_filter(np.real(data[1:100, 1:100]), sigma=0.5) + 1j * scipy.ndimage.gaussian_filter(
    #    np.imag(data[1:100, 1:100]), sigma=0.5)
    max_freq_bin_lif = 100
    data_lif_abs = np.abs(data_lif[1:max_freq_bin_lif, 1:max_freq_bin_lif])
    data_lif_angle = np.angle(data_lif[1:max_freq_bin_lif, 1:max_freq_bin_lif])
    max_freq_bin_pif = 500
    data_pif_abs = np.abs(data_pif[1:max_freq_bin_pif, 1:max_freq_bin_pif])
    data_pif_angle = np.angle(data_pif[1:max_freq_bin_pif, 1:max_freq_bin_pif])

    # plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    ax1.set_title("$|\chi_2^{LIF}(f_1, f_2)|$")
    pos = ax1.matshow(data_lif_abs, extent=[0, max_freq_bin_lif / 100, max_freq_bin_lif / 100, 0], vmin=0, vmax=300)
    fig.colorbar(pos, ax=ax1)

    ax2.set_title("$|\chi_2^{PIF}(f_1, f_2)|$")
    pos = ax2.matshow(data_pif_abs, extent=[0, max_freq_bin_pif / 100, max_freq_bin_pif / 100, 0], vmin=0, vmax=300)
    fig.colorbar(pos, ax=ax2)

    ax3.set_title("$\phi(\chi_2^{LIF}(f_1, f_2))$")
    pos = ax3.matshow(data_lif_angle, extent=[0, max_freq_bin_lif / 100, max_freq_bin_lif / 100, 0])
    fig.colorbar(pos, ax=ax3)

    ax4.set_title("$\phi(\chi_2^{PIF}(f_1, f_2))$")
    pos = ax4.matshow(data_pif_angle, extent=[0, max_freq_bin_pif / 100, max_freq_bin_pif / 100, 0])
    fig.colorbar(pos, ax=ax4)

    fig.savefig("img/suscept_nonlin_matrix_lif_pif.png")
    plt.show()


if __name__ == "__main__":
    main()
