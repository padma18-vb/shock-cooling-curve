import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.utils import io

import extinction
import pysynphot as S
import scipy.optimize as opt

rsun = 6.96e10
sigma = 5.67e-5


class supernova(object):
    def __init__(self, d):

        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [supernova(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, supernova(b) if isinstance(b, dict) else b)
        self.controls()

    def controls(self):
        self.assert_params()
        self.build_df()
        self.build_bandpass()

    def assert_params(self):
        self.a_v = self.ebv_mw * 3.1
        self.a_v_host = self.ebv_host * 3.1
        self.dsn = self.dist_sn * 3.086e+24
        self.model = None

        # input columns
        self.set_mag_colname()
        self.set_flt_colname()
        self.set_date_colname()
        self.set_magerr_colname()

        self.rmag_colname = 'RMAG'
        self.shift_date_colname = 'MJD_S'

        self.filt_to_wvl = {'w2': 1928, 'm2': 2246, 'w1': 2600, 'us': 3465, 'bs': 4392, 'vs': 5468, 'u': 3546,
                            'b': 4350.6, 'g': 4765.1, 'c': 5350., 'v': 5401.4, 'r': 6223.3, 'o': 6900., 'i': 7609.2,
                            'z': 8917, 'j': 12355, 'h': 16458, 'k': 21603}

        self.color_dict = {'w2': 'magenta', 'm2': 'cyan', 'w1': 'dodgerblue', 'u': 'purple', 'us': 'purple',
                           'bs': 'blue',
                           'vs': 'violet', 'b': 'blue',
                           'v': 'violet', 'r': 'red', 'o': 'orange', 'g': 'green', 'i': 'darkorange'}

    def display_file(self):
        return pd.read_csv(self.filename)

    def set_mag_colname(self, name='MAG'):
        self.mag_colname = name

    def set_magerr_colname(self, name ='MAGERR'):
        self.magerr_colname = name

    def set_date_colname(self, name='MJD'):
        self.date_colname = name

    def set_flt_colname(self, name='FLT'):
        self.flt_colname = name

    def build_df(self):
        self.data_all = pd.read_csv(self.filename)

        def add_red_mag(df, mag_col, flt_col):
            # makes no change to original data frame for future reference for user
            dfnew = df.copy()
            mags = np.array(dfnew[mag_col])
            filts = np.char.lower((np.array(dfnew[flt_col], dtype=str)))
            mag_to_filt = dict(zip(mags, filts))

            def reduce(mag):
                filt = mag_to_filt[mag]
                wvl = np.array([self.filt_to_wvl[filt]])
                m_dr = mag - extinction.fitzpatrick99(wvl, self.a_v, r_v=3.1, unit='aa')
                m_dr = m_dr - extinction.fitzpatrick99(wvl, self.a_v_host, r_v=3.1, unit='aa')
                return m_dr

            reduced_mags = np.array([reduce(mag) for mag in mags])
            dfnew[self.rmag_colname] = reduced_mags
            dfnew[flt_col] = [i.lower() for i in dfnew[flt_col]]

            return dfnew

        reduced_df = add_red_mag(self.data_all, self.mag_colname, self.flt_colname)
        reduced_df = reduced_df[reduced_df[self.date_colname] <= self.end_sc]
        reduced_df = reduced_df.assign(MJD_S=reduced_df[self.date_colname] - self.start_sn)
        self.reduced_df = reduced_df

    def get_filts(self):
        return pd.unique(self.reduced_df[self.flt_colname])

    def build_bandpass(self):
        self.swift_files = {}  # key: flt, value: array containing filter values
        self.bandpass_dict = {}  # key: flt, value: bandpass
        filts = self.get_filts()
        for flt in filts:
            try:
                self.swift_files[flt] = np.genfromtxt(f'./Swift_Filters/Swift_UVOT.{flt.upper()}.dat')
                arr = self.swift_files[flt]
                self.bandpass_dict[flt] = S.ArrayBandpass(arr[:, 0], arr[:, 1], name=flt)
            except:
                self.bandpass_dict[flt] = S.ObsBandpass(f'sdss,{flt}')
                pass
        # print(self.reduced_df, '\nbandpass keys',self.bandpass_dict.keys(),'\nswift file keys',self.swift_files.keys())

    def BB_lam(self, lam, T):
        from scipy.constants import h, k, c
        hh = h * 1e7
        kk = k * 1e7
        cc = c * 1e2
        lam_cm = 1e-8 * lam  # convert to cm
        rad = 2 * hh * cc ** 2 / (lam_cm ** 5 * (np.exp(hh * cc / (lam_cm * kk * T)) - 1))
        radd = rad / 1e8  # erg / s / cm^2 / Ang.
        return radd

    # only called when fitting occurs
    def mags_per_filter(self, times, Re, Me, ve=None, filtName=None):
        '''

        :param times: observation times
        :param Re: Radius of envelope in R_sun
        :param Me: Mass of envelope
        :param ve: velocity of propagation (how do we say this) how fast the wave is moving out
        :param filtName: filter name
        :return: array of all mags for specified times fitted assuming the a BB model according to specified filter
        '''

        bpname = filtName.lower()

        if ve is not None:
            R, T = self.model(times, Re, Me, ve)
        else:
            R, T = self.model(times, Re, Me)

        R = R.flatten()
        T = T.flatten()
        with io.capture_output() as captured:

            mags = np.array([])
            for i in range(len(T)):
                bb = S.BlackBody(T[i])
                bb.convert('flam')  # erg / s / cm^2 / AA
                l = np.linspace(np.min(bb.wave), np.max(bb.wave), 1000)
                norm = (bb.integrate(fluxunits='flam') ** -1.) * np.trapz(self.BB_lam(l, T[i]), l)
                bb_norm = bb * norm
                Flam = (np.pi / (self.dsn ** 2)) * (R[i] ** 2) * bb_norm  # ergs/ s / Ang.

                bandpass = self.bandpass_dict[bpname]
                obs = S.Observation(Flam, bandpass)
                mag = obs.effstim('abmag')
                mags = np.append(mags, mag)
                # return estimate of distance

        return mags

    def get_all_mags_without_v(self, times, Re, Me, of):
        filters = np.array(self.reduced_df['FLT'])
        # with io.capture_output() as captured:
        # mjd date, of - free parameter
        all_mags = np.array([])
        for i in range(len(times)):
            flt = filters[i]
            time = times[i] - of
            val = self.mags_per_filter(time, Re, Me, filtName=flt)

            all_mags = np.append(all_mags, val)
        return all_mags

    def get_all_mags_with_v(self, times, Re, Me, ve, of):
        '''
        Set
        :param times: Times at which data is available
        :param Re: Radius of envelope
        :param Me: Mass of envelope
        :param ve: Velocity of wave moving out
        :param of: Offset for observation time
        :return: array of all mags at specified time according to filter
        '''
        filters = np.array(self.reduced_df['FLT'])
        # with io.capture_output() as captured:
        # mjd date, of - free parameter
        all_mags = np.array([])
        for i in range(len(times)):
            flt = filters[i]
            time = times[i] - of
            val = self.mags_per_filter(time, Re, Me, ve, filtName=flt)

            all_mags = np.append(all_mags, val)
        return all_mags

    def simple_curve_fit(self, initial_parameters: tuple, lower_bounds: tuple, upper_bounds: tuple):
        if len(initial_parameters) == 3:
            func = self.get_all_mags_without_v
        else:
            func = self.get_all_mags_with_v
        phase_combo = np.array(self.reduced_df[self.shift_date_colname])
        mag_combo = np.array(self.reduced_df[self.mag_colname])
        emag_combo = np.array(self.reduced_df[self.magerr_colname])
        popt, pcov = opt.curve_fit(f=func, xdata=phase_combo, ydata=mag_combo, p0=initial_parameters,
                                   sigma=emag_combo, bounds=[lower_bounds, upper_bounds])
        sigma = np.sqrt(np.diag(pcov))

        self.fitted_params = popt
        self.fitted_errors = sigma

    def get_sc_RE(self):
        return self.fitted_params[0], self.fitted_errors[0]

    def get_sc_ME(self):
        return self.fitted_params[1], self.fitted_errors[1]

    def get_sc_VE(self):
        if len(self.fitted_params) > 3:
            return self.fitted_params[2], self.fitted_errors[2]
        else:
            print('Velocity is not a fitted paramter.')

    def get_sc_OF(self):
        if len(self.fitted_params) > 3:
            return self.fitted_params[3], self.fitted_errors[3]
        else:
            return self.fitted_params[2], self.fitted_errors[2]

    def write_to_file(self):
        

    def plot_given_parameters(self, Re, Me, ve=None, of=0, errorbar=None, shift=False):
        # print(f'Best radius = {Re}, err')
        # print(f'Best mass = {Me}')
        # if ve:
        #     print(f'Best velocity = {ve}')
        # print(f'Best offset = {of}')

        fig = plt.figure(figsize=(10, 10))
        unique_filts = self.get_filts()
        n = len(unique_filts)
        minmag = min(self.reduced_df[self.mag_colname]) - n // 2
        maxmag = max(self.reduced_df[self.mag_colname]) + n // 2
        t = np.linspace(0.01, max(self.reduced_df[self.shift_date_colname]) + 1, 100)

        def build_offset(n):
            offset = [i for i in range(-n // 2, n // 2)]
            return offset

        yerr = errorbar
        if shift:
            offset = build_offset(n)
        else:
            offset = [0] * len(unique_filts)

        # HARDCODED
        for flt in self.filt_to_wvl:
            if flt in unique_filts:
                filtered = self.reduced_df[self.reduced_df[self.flt_colname] == flt]
                times = np.array(filtered['MJD_S']) - of
                off = offset.pop(0)
                mag_all = np.array(filtered[self.mag_colname]) + off
                if errorbar != None:
                    yerr = np.array(filtered[errorbar])
                # print(flt)
                # DISCRETE
                if flt in self.color_dict:
                    plt.errorbar(x=times, y=mag_all, yerr=yerr, fmt='o', markersize=14, markeredgecolor='k',
                                 color=self.color_dict[flt], label=flt)
                else:
                    plt.errorbar(x=times, y=mag_all, yerr=yerr, fmt='o', markersize=14, markeredgecolor='k', label=flt)
                # CONTINUOUS

                vals = self.mags_per_filter(t, Re, Me, ve, filtName=flt) + off

                plt.plot(t, vals, color=self.color_dict[flt])
                plt.legend(loc='upper right', fontsize=20, ncol=2)
                plt.ylim(maxmag + off, minmag - off)
        plt.show()
        plt.savefig('test.png')
        return fig


