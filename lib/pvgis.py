import numpy as np
import pandas as pd
from datetime import datetime
from io import StringIO
import logging
import requests
from requests.exceptions import HTTPError
import pytz
import argparse
from matplotlib import pyplot as plt
from geopy.geocoders import Nominatim


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('in module %(name)s, in func %(funcName)s, '
                              '%(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
if not len(logger.handlers):
    logger.addHandler(stream_handler)
    logger.propagate = False


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PV parameters to PVGIS')
        self.arg_list = ['lat', 'lon', 'file', 'city']
        self.coordinate, self.city, self.file = False, False, False

    def parse_args(self):
        self.parser.add_argument('--lat', type=float, required=False,
                                 default=None,
                                 help='latitude of location')
        self.parser.add_argument('--lon', type=float, required=False,
                                 default=None,
                                 help='longitude of location')
        self.parser.add_argument('--file', type=str, required=False,
                                 default=None,
                                 help='file name for saving output data')
        self.parser.add_argument('--city', type=str, required=False,
                                 default=None,
                                 help='city name to locate PV')
        return self.parser.parse_args()

    def get_args(self):
        _args = self.parse_args()
        if _args.lon is not None and _args.lat is not None:
            self.coordinate = True
            print(f'lon = {_args.lon} \t lat = {_args.lat}')
        elif _args.city is not None:
            self.city = True
            print(f'city = {_args.city}')
        if _args.file is not None:
            self.file = True
        else:
            logger.error('Bad args were provided.')
        return _args


def parse_args():
    parser = argparse.ArgumentParser(
        description='PV parameters to PVGIS')
    parser.add_argument('--lat', type=float, required=False,
                        default=None,
                        help='latitude of location')
    parser.add_argument('--lon', type=float, required=False,
                        default=None,
                        help='longitude of location')
    parser.add_argument('--file', type=str, required=False,
                        default='pvgis_output.csv',
                        help='file name for saving output data')
    parser.add_argument('--city', type=str, required=False,
                        default='Prague',
                        help='city name to locate PV')
    parser.add_argument('--f_type', type=str, required=False,
                        default='excel',
                        help='Saving file with this type')
    parser.add_argument('--plot', type=bool, required=False,
                        default=False,
                        help='plot result')
    return parser.parse_args()


class Writer:
    def __init__(self, file_name):
        self.file_name = file_name

    # @property
    # def file_name(self):
    #     return self._file_name
    #
    # @file_name.setter
    # def file_name(self, file_name):
    #     folder = os.path.split(file_name)[0]
    #     if folder:
    #         if not os.path.exists(folder):
    #             os.mkdir(folder)
    #
    #     if os.path.exists(folder):
    #         self._file_name = file_name
    #     else:
    #         self._file_name = None
    #         logger.error('Folder: {} does not exists.'.format(folder))

    # @file_name.getter
    def write_csv(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            df.to_csv(self.file_name)
            logger.info(f'df was successfully written to {self.file_name}')
        else:
            logger.error(f'df passed if not pd.DataFrame, but {type(df)}')

    def write_excel(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            df.to_excel(self.file_name)
            logger.info(f'df was successfully written to {self.file_name}')
        else:
            logger.error(f'df passed if not pd.DataFrame, but {type(df)}')


class CityParser:
    def __init__(self, city: str):
        self.city = city
        locator = Nominatim(user_agent='Feramat_Cybernetics_PVGIS_parser')
        self.location = locator.geocode(city)

    def get_coordinates(self):
        return self.location.latitude, self.location.longitude


class PVGIS:
    def __init__(
            self, city=None, lat=50.1, lon=14.46, out_format='csv',
            start_year=2015, end_year=2015, slope=0, azimuth=0, peak_power=1, sys_loss=14
    ):
        self.city = city
        if self.city is not None:
            _lat, _lon = CityParser(self.city).get_coordinates()
        else:
            _lat, _lon = lat, lon
        self.lat = _lat
        self.lon = _lon
        self.out_format = out_format
        self.start_year = start_year
        self.end_year = end_year
        self.peak_power = peak_power
        self.sys_loss = sys_loss

    @staticmethod
    def technology_mapper(tech):
        switcher = dict(
            poly='crystSi',
            mono='mono'
        )
        return switcher.get(tech)

    def _create_url(self):
        url_base = 'https://re.jrc.ec.europa.eu/api/seriescalc?' \
                   'lat={lat}&lon={lon}&raddatabase=PVGIS-SARAH&browser=0&' \
                   'outputformat={out_format}&userhorizon=&usehorizon=0&angle=&aspect=&' \
                   'startyear=2015&endyear=2015&mountingplace=free&' \
                   'optimalinclination=0&optimalangles=1&js=1&' \
                   'select_database_hourly=PVGIS-SARAH&' \
                   'hstartyear={start_year}&hendyear={end_year}&trackingtype=0&' \
                   'hourlyoptimalangles=1&pvcalculation=1&pvtechchoice={pv_tech}&peakpower={peak_power}&loss={loss}'. \
            format(
                lat=self.lat, lon=self.lon, out_format=self.out_format,
                start_year=self.start_year, end_year=self.end_year,
                pv_tech='crystSi', peak_power=self.peak_power, loss=self.sys_loss
                )
        return url_base

    def make_request(self):
        response = None
        try:
            response = requests.get(self._create_url())
            response.raise_for_status()
        except HTTPError as http_err:
            logger.error(f'HTTP error occurred: {http_err}, status: {response.status_code}')
        except Exception as err:
            logger.error(f'Other error occurred: {err}')
        else:
            dt = datetime.now(pytz.timezone('Europe/Prague'))
            logger.info(
                f"Success connection at time: {dt.strftime(format='%Y-%m-%d %H:%M:%S %z')}!"
            )
        return response

    @staticmethod
    def create_df(data, year=2019):
        df_pv = pd.read_csv(data, skiprows=10, ).iloc[:8760, :]
        df_pv.columns = [col.lower() for col in df_pv.columns]
        df_pv.time = pd.to_datetime(df_pv.time, format='%Y%m%d:%H%M')
        df_pv.time = df_pv.time.dt.round('H')
        df_pv.index = df_pv.time
        df_pv.index = df_pv.index.map(lambda t: t.replace(year=year))

        df_pv = df_pv.rename(mapper={
            'p': "pv"
        }, axis=1)
        df_pv.pv = df_pv.pv.astype('float')
        cols_to_drop = [col for col in df_pv.columns if col not in ['pv']]
        df_pv.drop(cols_to_drop, axis=1, inplace=True)
        # # convert to kW
        # df_pv.pv_power = df_pv.pv_power.astype(float)
        df_pv.pv *= 1e-3
        return df_pv

    def parse(self):
        response = self.make_request()
        df = self.create_df(StringIO(response.text))
        checker = self._check_coordinates(self._parse_coordinates_api(response.text), (self.lat, self.lon))
        self._parse_coordinates_api(response.text)
        self.get_df_info(df)
        if checker:
            logger.info('Coordinates from API are close to lat and lon.')
        else:
            logger.info('Coordinates from API are NOT close to lat and lon!!!')
        return df

    @staticmethod
    def _parse_coordinates_api(response_text: str):
        coordinate_list = response_text.split('\t')[1:3]
        coordinates = (float(coordinate_list[0].split()[0]), float(coordinate_list[1].split()[0]))
        return coordinates

    @staticmethod
    def _check_coordinates(coordinate_data: tuple, coordinate_api: tuple):
        tolerance_array = np.ones(2, dtype=float) * 0.05
        diff_array = np.abs(np.array(coordinate_data) - np.array(coordinate_api))
        return any(diff_array <= tolerance_array)

    @staticmethod
    def get_df_info(df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            print(f'df.shape = {df.shape}')
            print(f'df.isna().any() = {df.pv.isna().any()}')
            print(f'df.pv.sum() = {df.pv.sum():.2f} kWh.a\N{HYPHEN}\N{SUPERSCRIPT ONE}')
            print('-' * 50)
            print('df.head():')
            print(df.head())


def main(lat, lon, file_name, city):
    pv = PVGIS()
    df = pv.parse()
    print(df.shape)
    print(df.sum() / 1e3)
    plt.plot(df.pv)
    plt.show()

    # writer = Writer(os.path.join('data', 'file.csv'))
    # print()


# if __name__ == '__main__':
#     args = parse_args()
#     main(args.lat, args.lon, args.file, args.city)
