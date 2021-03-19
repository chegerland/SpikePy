import matplotlib.pyplot as plt
import numpy as np
import csv


def main():

    path = "./../Spike/data/Firing_rate/lifac_two_sigs_raster.csv"

    datafile = open(path, 'r')
    datareader = csv.reader(datafile)
    data = []
    for row in datareader:
        # I split the input string based on the comma separator, and cast every elements into a float
        data.append([float(elem) for elem in row])

    plt.eventplot(data)
    plt.xlim([1, 100])
    plt.show()


if __name__ == "__main__":
    main()
