from lib.pvgis import PVGIS, Writer, parse_args
import os
from matplotlib import pyplot as plt


def main(lat, lon):
    pv = PVGIS(lat=lat, lon=lon).parse()
    Writer(os.path.join('data', 'prague.csv')).write(pv)
    plt.plot(pv)


if __name__ == '__main__':
    args = parse_args()
    main(args.lat, args.lon)

