from lib.pvgis import PVGIS, Writer
import os
from matplotlib import pyplot as plt


def main():
    pv = PVGIS().parse()
    Writer(os.path.join('data', 'prague.csv')).write(pv)
    plt.plot(pv)


if __name__ == '__main__':
    main()
