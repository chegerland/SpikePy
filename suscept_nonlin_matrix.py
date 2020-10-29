#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

def main():
    #data = np.genfromtxt("../Spike/data/NonlinMatrix/lif_suscept_matrix.csv", delimiter=',')
    data = np.genfromtxt("../Spike/data/NonlinMatrix/lif_Ne5_suscept_matrix.csv", delimiter=',')
    #data = np.genfromtxt("../Spike/data/NonlinMatrix/lifac_suscept_matrix.csv", delimiter=',')
    #data = np.genfromtxt("../Spike/data/NonlinMatrix/lif_reduced_suscept_matrix.csv", delimiter=',')

    #data = scipy.ndimage.uniform_filter(data[1:100, 1:100], size=2, mode='constant')
    #data = scipy.ndimage.gaussian_filter(data[1:1000, 1:1000], sigma=0.5)
    #data = abs(data)
    data = abs(data[1:100, 1:100])

    fig, ax = plt.subplots()
    ax.set_title("LIFAC")
    pos = ax.matshow(data)
    fig.colorbar(pos, ax=ax)

    #fig.savefig("suscept_nonlin_matrix_lifac_unfiltered.png")
    plt.show()


if __name__ == "__main__":
    main()
