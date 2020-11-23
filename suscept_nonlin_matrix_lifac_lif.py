#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import neurons

from math import pi


def main():
    # analytics
    lifac = neurons.LIFAC(3.5, 0.1, 0.1, 10.0)
    steps = 100
    chi_2_lifac = np.zeros(shape=(steps, steps), dtype=complex)
    chi_2_lif = np.zeros(shape=(steps, steps), dtype=complex)
    f = np.linspace(0.0, 3.5, num=steps)

    for i in range(steps):
        for j in range(steps):
            chi_2_lifac[i, j] = lifac.susceptibility_2(2.0 * pi * f[i], 2.0 * pi * f[j])
            chi_2_lif[i, j] = lifac.lif.susceptibility_2(2.0 * pi * f[i], 2.0 * pi * f[j])

    # data
    # data = np.genfromtxt("../Spike/data/NonlinMatrix/lif_suscept_matrix.csv", delimiter=',')
    # data = np.genfromtxt("../Spike/data/NonlinMatrix/lif_Ne5_suscept_matrix.csv", delimiter=',')
    data = np.genfromtxt("../Spike/data/NonlinMatrix/lifac_suscept_matrix.csv", delimiter=',', dtype=np.complex128)
    #data = np.genfromtxt("../Spike/data/LIF_Check/lif_suscept_matrix.csv", delimiter=',', dtype=np.complex128)

    max_freq_bin = 350
    data = scipy.ndimage.gaussian_filter(np.real(data[1:max_freq_bin, 1:max_freq_bin]), sigma=0.5) + 1j * scipy.ndimage.gaussian_filter(
        np.imag(data[1:max_freq_bin, 1:max_freq_bin]), sigma=0.5)
    data_abs = np.abs(data)
    data_angle = np.angle(data)
    #data_abs = np.abs(data[1:max_freq_bin, 1:max_freq_bin])
    #data_angle = np.angle(data[1:max_freq_bin, 1:max_freq_bin])

    # plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 20))
    ax1.set_title("$|\chi_2^{LIFAC(ana)}(f_1, f_2)|$")
    pos1 = ax1.matshow(np.abs(chi_2_lifac), extent=[0, max_freq_bin / 100, max_freq_bin / 100, 0], vmin=0, vmax=0.7)
    fig.colorbar(pos1, ax=ax1)

    ax2.set_title("$|\chi_2^{LIFAC(num)}(f_1, f_2)|$")
    pos2 = ax2.matshow(data_abs, extent=[0, max_freq_bin / 100, max_freq_bin / 100, 0], vmin=0, vmax=0.7)
    fig.colorbar(pos2, ax=ax2)

    ax3.set_title("$|\chi_2^{LIF}(f_1, f_2)|$")
    pos3 = ax3.matshow(np.abs(chi_2_lif), extent=[0, max_freq_bin / 100, max_freq_bin / 100, 0], vmin=0, vmax=0.7)
    fig.colorbar(pos3, ax=ax3)

    ax4.set_title("$\phi(\chi_2^{LIFAC(ana)}(f_1, f_2))$")
    pos4 = ax4.matshow(-np.angle(chi_2_lifac), extent=[0, max_freq_bin / 100, max_freq_bin / 100, 0])
    cbar4 = fig.colorbar(pos4, ax=ax4, ticks=[-pi, -pi/2, 0, pi/2, pi])
    cbar4.ax.set_yticklabels(['$-\pi$', '$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$', '$\pi'])

    ax5.set_title("$\phi(\chi_2^{LIFAC(num)}(f_1, f_2))$")
    pos5 = ax5.matshow(data_angle, extent=[0, max_freq_bin / 100, max_freq_bin / 100, 0])
    cbar5 = fig.colorbar(pos5, ax=ax5, ticks=[-pi, -pi/2, 0, pi/2, pi])
    cbar5.ax.set_yticklabels(['$-\pi$', '$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$', '$\pi'])

    ax6.set_title("$\phi(\chi_2^{LIF}(f_1, f_2))$")
    pos6 = ax6.matshow(-np.angle(chi_2_lif), extent=[0, max_freq_bin / 100, max_freq_bin / 100, 0])
    cbar6 = fig.colorbar(pos6, ax=ax6, ticks=[-pi, -pi/2, 0, pi/2, pi])
    cbar6.ax.set_yticklabels(['$-\pi$', '$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$', '$\pi'])

    fig.savefig("img/suscept_nonlin_matrix_lifac.png")
    plt.show()


if __name__ == "__main__":
    main()
