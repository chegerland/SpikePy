import os
import numpy as np
from math import pi


def calculate_linear_suscept(file_path, f_min, f_max, steps, susceptibility):
    # check if there is already a file
    if os.path.isfile(file_path):
        chi_1 = np.genfromtxt(file_path, delimiter=',', dtype=complex)
        print("Reading matrix from file", file_path)
    else:
        print("Calculating matrix...")

        # define resolution
        f = np.linspace(f_min, f_max, num=steps)
        chi_1 = np.zeros(shape=(steps), dtype=complex)

        # calculate minimal matrix
        for i in range(steps):
            print("Step", i, "of", steps)
            chi_1[i] = susceptibility(2.0 * pi * f[i])

        # save the minimal matrix
        print("Saving matrix to file", file_path)
        np.savetxt(file_path, chi_1, delimiter=',')

    return chi_1


def calculate_novikov_suscept(file_path_1, file_path_2, file_path_3, f_min, f_max, steps, susceptibility_1, susceptibility_2):
    # check if there is already a file
    if os.path.isfile(file_path_2) and os.path.isfile(file_path_2) and os.path.isfile(file_path_3):
        chi_1 = np.genfromtxt(file_path_1, delimiter=',', dtype=complex)
        chi_2_diag = np.genfromtxt(file_path_2, delimiter=',', dtype=complex)
        chi_2_antidiag = np.genfromtxt(
            file_path_3, delimiter=',', dtype=complex)
        print("Reading linear suscept from file", file_path_1)
        print("Reading diagonal nonlinear suscept from file", file_path_2)
        print("Reading antidiagonal nonlinear suscept from file", file_path_3)
    else:
        print("Calculating suscepts...")

        # define resolution
        f = np.logspace(f_min, f_max, num=steps)
        chi_1 = np.zeros(shape=(steps), dtype=complex)
        chi_2_diag = np.zeros(shape=(steps), dtype=complex)
        chi_2_antidiag = np.zeros(shape=(steps), dtype=complex)

        # calculate minimal matrix
        for i in range(steps):
            print("Step", i, "of", steps)
            chi_1[i] = susceptibility_1(2.0 * pi * f[i])
            chi_2_diag[i] = susceptibility_2(2.0 * pi * f[i], 2.0 * pi * f[i])
            chi_2_antidiag[i] = susceptibility_2(
                2.0 * pi * f[i], -2.0 * pi * f[i])

        # save the minimal matrix
        print("Saving suscepts to file", file_path_1, file_path_2, file_path_3)
        np.savetxt(file_path_1, chi_1, delimiter=',')
        np.savetxt(file_path_2, chi_2_diag, delimiter=',')
        np.savetxt(file_path_3, chi_2_antidiag, delimiter=',')

    return chi_1, chi_2_diag, chi_2_antidiag


def calculate_analytic_matrix(file_path, f_min, f_max, steps, susceptibility):
    """
    Calculates the second order susceptibility. Uses cashing, so if the matrix has already been calculated we simply
    load the csv file.

    :param file_path: Path to .csv file
    :param f_min: Minimal frequency
    :param f_max: Maximum frequency
    :param steps: Number of steps (resolution)
    :param susceptibility: Method with which to calculate the matrix
    :return: The (second order) susceptibility matrix
    """
    # check if minimal matrix calculated
    if os.path.isfile(file_path):
        chi_2 = np.genfromtxt(file_path, delimiter=',', dtype=complex)
        print("Reading matrix from file", file_path)
    else:
        print("Calculating matrix...")

        # define resolution
        f = np.linspace(f_min, f_max, num=steps)
        chi_2 = np.zeros(shape=(steps, steps), dtype=complex)

        # calculate minimal matrix
        for i in range(steps):
            print("Step", i, "of", steps)
            for j in range(steps):
                if i <= j:
                    chi_2[i][j] = susceptibility(
                        2.0 * pi * f[i], 2.0 * pi * f[j])
                else:
                    chi_2[i][j] = susceptibility(
                        2.0 * pi * f[i], -2.0 * pi * f[j])

        # save the minimal matrix
        print("Saving matrix to file", file_path)
        np.savetxt(file_path, chi_2, delimiter=',')

    return chi_2


def create_full_matrix(chi_2_numeric):
    """
    Creates all areas in the frequency plot from the reduced numeric matrix.
    :param chi_2_numeric: The numeric matrix
    :return: The full matrix
    """
    steps = chi_2_numeric.shape[0]

    # matrix for upper right corner
    chi_2_ur = chi_2_numeric.copy()
    for i in range(steps):
        for j in range(steps):
            if i >= j:
                chi_2_ur[i][j] = chi_2_ur[j][i]

    # matrix for lower left corner
    chi_2_ll = np.conj(np.flip(chi_2_ur))

    # matrix for upper left corner
    chi_2_ul = chi_2_numeric.copy()
    for i in range(steps):
        for j in range(steps):
            if i <= j:
                chi_2_ul[i][j] = np.conj(chi_2_numeric[j][i])
    chi_2_ul = np.flip(chi_2_ul, 1)

    # matrix for lower right corner
    chi_2_lr = np.conj(np.flip(chi_2_ul))

    # put all domains together
    chi_2 = np.zeros(shape=(2 * steps, 2 * steps), dtype=complex)
    for i in range(2 * steps):
        for j in range(2 * steps):
            # upper right
            if i >= steps and j >= steps:
                chi_2[i][j] = chi_2_ur[i - steps][j - steps]

            # upper left
            if i >= steps > j:
                chi_2[i][j] = chi_2_ul[i - steps][j - steps]

            # lower left
            if i < steps and j < steps:
                chi_2[i][j] = chi_2_ll[i][j]

            # lower right
            if i < steps <= j:
                chi_2[i][j] = chi_2_lr[i][j - steps]

    return chi_2


def plot_absolute_value(fig, axis, data, **kwargs):
    """
    Plots the absolute value of data. This method is meant to be used for the plotting of the second order
    susceptibility matrix.
    """
    # plot the matrix with a colorbar
    pos = axis.matshow(data, **kwargs, origin='lower')
    fig.colorbar(pos, ax=axis)

    # plot lines for orientation
    axis.axvline(x=0, color='black')
    axis.axhline(y=0, color='black')
    axis.plot(axis.get_xlim(), axis.get_ylim(), linestyle="--", color="white")


def plot_complex_angle(fig, axis, data, **kwargs):
    """
    Plots the complex angle of data. This method is meant to be used for the plotting of the second order
    susceptibility matrix.
    """
    # plot the matrix with a colorbar,
    pos = axis.matshow(data, **kwargs, vmin=-pi, vmax=pi, origin='lower')
    cbar = fig.colorbar(pos, ax=axis, ticks=[-pi, -pi / 2, 0, pi / 2, pi])
    cbar.ax.set_yticklabels(
        ['$-\\pi$', '$-\\frac{\\pi}{2}$', '0', '$\\frac{\\pi}{2}$', '$\\pi$'])

    # plot lines for orientation
    axis.axvline(x=0, color='black')
    axis.axhline(y=0, color='black')
    axis.plot(axis.get_xlim(), axis.get_ylim(), linestyle="--", color="white")
