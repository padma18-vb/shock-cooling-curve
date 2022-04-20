import matplotlib.pyplot as plt
import numpy as np
from supernova import supernova
import argparse
import configparser

rsun = 6.96e10
sigma = 5.67e-5


class SW_RSG(supernova):
    def __init__(self, d):
        super().__init__(d)
        self.name = 'Sapir Waxman Red Super Giant'

    def L_SW_rsg(self, t, Re, Me, vss):
        M = self.mcore + Me
        k = 0.2 / 0.34  # 1.
        fp = (Me / self.mcore) ** 0.5  # n=3/2
        vs = (vss * 1e9) / (10 ** 8.5)
        # R = Re/1e13
        L = 1.88 * 1e42 * (((vs * (t ** 2.)) / (fp * M * k)) ** (-0.086)) * (vs ** 2. * Re / k) * np.exp(
            -(1.67 * t / (19.5 * (k * Me * vs ** -1) ** 0.5)) ** 0.8)
        T = 2.05 * 1e4 * (((vs ** 2. * t ** 2.) / (fp * M * k)) ** 0.027) * ((Re ** 0.25) / (k ** 0.25)) * (t ** -0.5)
        R = (L / (4. * np.pi * (T ** 4.) * sigma)) ** 0.5
        return np.array([R]), np.array([T])

    def set_model(self):
        self.model = self.L_SW_rsg

    def get_fitted_values(self):
        self.set_model()
        self.simple_curve_fit((2.,0.5, 2., 0.01), (0.01, 0.01, 0.01, 0.001), (10., 10. , 10., 1.0))
        print(f'{self.name} model simple curve fitted values:')

        print(f'Re = {self.get_sc_RE()[0] * 1e13 / rsun} +/- {self.get_sc_RE()[1] * 1e13 / rsun} R_Sun')
        print(f'Me = {self.get_sc_ME()[0]} +/- {self.get_sc_ME()[1]}')
        print(f'Ve = {self.get_sc_VE()[0] * 1e9 * 1e-5} +/- {self.get_sc_VE()[1] * 1e9 * 1e-5} km/s')
        print(f'Offset = {self.get_sc_OF()[0]} +/- {self.get_sc_OF()[1]}')

    def plot(self):
        self.plot_given_parameters(self.get_sc_RE()[0], self.get_sc_ME()[0], ve = self.get_sc_VE()[0], of=self.get_sc_OF()[0], shift=True)


class SW_BSG(supernova):
    def __init__(self, d):
        super().__init__(d)
        self.name = 'Sapir-Waxman Blue Super Giant'

    def L_SW_bsg(self, t, Re, Me, vss):
        M = self.mcore + Me
        k = 0.2 / 0.34  # 1.
        fp = (Me / self.mcore) * 0.08  # n=3
        vs = (vss * 1e9) / (10 ** 8.5)
        # R = Re/1e13
        L = 1.66 * 1e42 * (((vs * (t ** 2.)) / (fp * M * k)) ** (-0.175)) * (vs ** 2. * Re / k) * np.exp(
            -((4.57 * t) / (19.5 * (k * Me * (vs ** -1.)) ** 0.5)) ** (0.73))
        T = 1.96 * 1e4 * (((vs ** 2. * t ** 2.) / (fp * Me * k)) ** 0.016) * ((Re ** 0.25) / (k ** 0.25)) * (t ** -0.5)
        R = (L / (4. * np.pi * (T ** 4.) * sigma)) ** 0.5
        return np.array([R]), np.array([T])

    def set_model(self):
        self.model = self.L_SW_bsg

    def get_fitted_values(self):
        self.set_model()
        self.simple_curve_fit((0.5, 2., 2., 0.04), (0.001, 0.01, 0.01, 0.01), (3., 10., 10., 1.0))
        print(f'{self.name} model simple curve fitted values:')

        print(f'Re = {self.get_sc_RE()[0] * 1e13 / rsun} +/- {self.get_sc_RE()[1] * 1e13 / rsun} R_Sun')
        print(f'Me = {self.get_sc_ME()[0]} +/- {self.get_sc_ME()[1]}')
        print(f'Ve = {self.get_sc_VE()[0] * 1e9 * 1e-5} +/- {self.get_sc_VE()[1] * 1e9 * 1e-5} km/s')
        print(f'Offset = {self.get_sc_OF()[0]} +/- {self.get_sc_OF()[1]}')

    def plot(self):
        self.plot_given_parameters(self.get_sc_RE()[0], self.get_sc_ME()[0], ve = self.get_sc_VE()[0], of=self.get_sc_OF()[0], shift=True)

# class SW_RSG(supernova):
#     def __init__(self, d):
#         super().__init__(d)
#

class PIRO_2015(supernova):
    def __init__(self, d):
        super().__init__(d)
        self.set_model()
        self.name = 'Piro 2015'


    def L_P15(self, t, Re, Me, k=0.2):
        def tp(Esn, Mc, Me, k):
            return 0.9 * ((k / 0.34) ** 0.5) * ((Esn) ** -0.25) * ((Mc) ** 0.17) * ((Me / (0.01)) ** 0.57)  # days

        def Ee(Esn, Mc, Me):
            return 4e49 * (Esn) * ((Mc) ** -0.7) * ((Me / (0.01)) ** 0.7)

        def ve(Esn, Mc, Me):
            return 1e5 * (86400.0) * (2e9) * ((Esn) ** 0.5) * ((Mc) ** -0.35) * ((Me / (0.01)) ** -0.15)  # cm/d

        te = (Re * rsun) / ve(self.bestek3, self.mcore, Me)
        L = ((te * Ee(self.bestek3, self.mcore, Me)) / tp(self.bestek3, self.mcore, Me, k) ** 2.) * \
            np.exp((-t * (t + 2. * te)) / (2. * tp(self.bestek3, self.mcore, Me, k) ** 2.))

        R = (Re * rsun) + (ve(self.bestek3, self.mcore, Me) * t / (86400.0))

        R = np.array([R])

        T = (L / (4. * np.pi * (R ** 2.) * sigma)) ** 0.25

        T = np.array([T])
        # print(R, T)

        return R, T

    def set_model(self):
        self.model = self.L_P15

    def get_fitted_values(self, initial = (100, 0.01, 0.1), lower_bounds = (0.1, 0.0001, 0.0), upper_bounds =(500.0, 1.0, 2.0)):
        self.set_model()
        #allow
        self.simple_curve_fit(initial, lower_bounds, upper_bounds)
        print(f'{self.name} model simple curve fitted values:')
        print(f'Re = {self.get_sc_RE()[0]} +/- {self.get_sc_RE()[1]}')
        print(f'Me = {self.get_sc_ME()[0]} +/- {self.get_sc_ME()[1]}')
        print(f'Offset = {self.get_sc_OF()[0]} +/- {self.get_sc_OF()[1]}')


    def f_lam(self, lam, R, T):
        return (np.pi / (self.dsn ** 2)) * (R ** 2) * (self.BB_lam(lam, T))  # ergs/ s / Ang.

    def plot(self):
        self.plot_given_parameters(self.get_sc_RE()[0], self.get_sc_ME()[0], of=self.get_sc_OF()[0], shift=True)
        #print(f)
        #print('whats in f')
        #f.show()
        #f.savefig('test.png')

def parse_args():
    """
    Handle the command line arguments.
    Returns:
    Output of argparse.ArgumentParser.parse_args.
    """

    parser = argparse.ArgumentParser(description='Accepting config file containing supernova parameters and '
                                                 'data file.')
    # parser.add_argument('-m', '--model', dest='model', type=str,
    #                     help='Name of model you want to apply. Options:'
    #                          '1. Piro 2015\n'
    #                     '2. Sapir-Waxman 2017 BSG\n'
    #                     '3. Sapir-Waxman 2017 RSG\n'
    #                     '4. Piro 2020\n'
    #                     'Further notes: Models 2 and 3 vary in their polytropic indices (BSG; n=3, RSG; n=3/2)')

    parser.add_argument('-c', '--config', dest='config', type=str,
                        help='Config file containing the following parameters: '
                             '1. Start time of initial shock cooling curve (MJD), \n'
                             '2. End time of initial peak (MJD), \n'
                             '3. Amount of dust in host galaxy, \n'
                             '4. Amount of dust in MILKY WAY, \n'
                             '5. Mass of core in solar masses, \n'
                             '6. SN kinetic energy in units of 10^51 erg, \n'
                             '7. Distance to supernova, \n'
                             '8. Name of input .csv file.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    command_line_args = parse_args()
    conf = command_line_args.config
    config = configparser.ConfigParser()
    config.read(conf)
    details_dict = {}
    for key, val in config.items('DEFAULT'):
        if key != 'filename':
            details_dict[key] = float(val)
        else:
            details_dict[key] = val
            assert val[-3:] == 'csv', 'File must be .csv'

    

