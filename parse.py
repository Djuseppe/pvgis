from lib.pvgis import PVGIS, Writer, parse_args
from matplotlib import pyplot as plt


def main(lat, lon, file, city):
    pv = PVGIS(lat=lat, lon=lon, city=city).parse()
    if city:
        file = city + '.csv'
    Writer(file).write(pv)
    # plt.plot(pv)
    # plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args.lat, args.lon, args.file, args.city)

