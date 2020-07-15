from lib.pvgis import PVGIS, Writer, parse_args
from matplotlib import pyplot as plt


def main(lat, lon, file, city, f_type, plot):
    pv = PVGIS(lat=lat, lon=lon, city=city).parse()
    if city:
        file = city + '.csv'
    if f_type == 'csv':
        Writer(file).write_csv(pv)
    elif f_type == 'excel':
        file = city + '.xls'
        Writer(file).write_excel(pv)
    if plot:
        plt.plot(pv)
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args.lat, args.lon, args.file, args.city, args.f_type, args.plot)

