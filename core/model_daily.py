import numpy as np
import numba
import pystan
import pandas as pd
import h5py
from ftplib import FTP, error_perm
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pickle
import time
from netCDF4 import Dataset
import seaborn as sns
import warnings
import calendar
# warnings.filterwarnings("ignore")

sns.set_style('whitegrid')
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":12, "axes.labelsize":12})


class cultivarModel:

    def __init__(self, region_lats, region_longs, years, cultivar, sow_month, ripe_days, stan_model, yields,
                 region_tol=0.25, n_gf=100, weather=('temperature', 'rainfall'), metric='yield', metric_units='t/Ha'):
        self.reg_lats = region_lats
        self.reg_longs = region_longs
        self.reg_yrs = years
        self.sow_months = sow_month
        self.ripe_days = ripe_days
        self.reg_keys = self.get_day_keys()
        self.yield_data = yields
        self.cult = cultivar
        self.model = stan_model
        self.n_gf = n_gf
        self.tol = region_tol
        self.wthr_dict = {}
        self.wthr_anom_dict = {}
        self.mean_params = {}
        self.fit = None
        self.samples = None
        self.reg_pred = np.zeros_like(self.yield_data)
        self.resi = None
        self.weather = weather
        self.metric = metric
        self.metric_units = metric_units

        # Open file
        if len(self.wthr_anom_dict.keys()) == 0:
            self.temp_anom = self.get_weather_anomaly(weather[0])
            self.precip_anom = self.get_weather_anomaly(weather[1])
        # self.sun_anom = self.get_weather_anomaly(hdf, 'sun')
        # self.modeldata = {'n_regions': len(self.reg_lats), 'n_years': self.reg_yrs.shape[1],
        #                   'd_temp': self.wthr_dict['temperature'], 'd_precip': self.wthr_dict['rainfall'],
        #                   'd_sun': self.wthr_dict['sun'],
        #                   'd_yields': self.yield_data,
        #                   'n_gf': n_gf, 'temp': np.linspace(-10, 40, n_gf),
        #                   'precip': np.linspace(0, 250, n_gf),
        #                   'sun': np.linspace(0, 300, n_gf)}
        # self.modeldata = {'n_regions': len(self.reg_lats), 'n_years': self.reg_yrs.shape[1],
        #                   'd_temp': self.wthr_dict[weather[0]], 'd_precip': self.wthr_dict[weather[1]],
        #                   'd_yields': self.yield_data,
        #                   'n_gf': n_gf, 'temp': np.linspace(-10, 40, n_gf),
        #                   'precip': np.linspace(0, 250, n_gf)}
        self.modeldata = {'n_regions': len(self.reg_lats), 'n_years': self.reg_yrs.shape[1],
                          'd_temp': self.temp_anom, 'd_precip': self.precip_anom,
                          'd_yields': self.yield_data,
                          'n_gf': n_gf, 'temp': np.linspace(-10, 40, n_gf),
                          'precip': np.linspace(0, 250, n_gf)}

    @staticmethod
    def extract_region(lat, long, region_lat, region_long, weather, tol):

        # Get the boolean array for points within tolerence
        bool_cond = np.logical_and(np.abs(lat - region_lat) < tol, np.abs(long - region_long) < tol)

        # Get the extracted region
        ex_reg = weather[bool_cond]

        # Remove any nan and set them to 0 these correspond to ocean
        ex_reg = ex_reg[np.where(weather[bool_cond] < 10E8)]

        if ex_reg.size == 0:
            return 0
        else:
            return np.mean(ex_reg)

    @staticmethod
    def gauss3d(norm, t, mu_t, sig_t, p, mu_p, sig_p, s, mu_s, sig_s, rho_tp, rho_ts, rho_ps):
        
        dy = 0
        for mon in range(0, 366):
            dy += norm * np.exp(-(0.5 * 1 / (1 - np.square(rho_tp) - np.square(rho_ts) - np.square(rho_ps)
                                             + 2 * rho_tp * rho_ts * rho_ps))
                                * (np.square((t[mon] - mu_t) / sig_t)
                                   + np.square((p[mon] - mu_p) / sig_p) + np.square((s[mon] - mu_s) / sig_s)
                                + 2 * ((t[mon] - mu_t) * (p[mon] - mu_p) * (rho_ts * rho_ps - rho_tp) / (sig_t * sig_p)
                                       + (t[mon] - mu_t) * (s[mon] - mu_s) * (rho_tp * rho_ts - rho_ps)
                                       / (sig_t * sig_s) + (p[mon] - mu_p) * (s[mon] - mu_s)
                                       * (rho_tp * rho_ts - rho_ps) / (sig_s * sig_p))))

        return dy
    
    @staticmethod
    def gauss2d(norm, t, mu_t, sig_t, p, mu_p, sig_p, rho, dy=0):

        for mon in range(0, 366):
            if t[mon] == 0:
                continue
            else:
                dy += norm * np.exp(-(1/(2 - 2*np.square(rho)))*(np.square( (t[mon] - mu_t) / sig_t) +
                                                                 np.square( (p[mon] - mu_p)/sig_p)
                                                                 - 2*rho*(t[mon]-mu_t)*(p[mon] - mu_p)/(sig_t*sig_p)))
        return dy

    @staticmethod
    def gauss2d_country(norm, t, mu_t, sig_t, p, mu_p, sig_p, rho, dy=0):

        for mon in range(0, 366):
            dy += norm * np.exp(-(1/(2 - 2*np.square(rho)))*(np.square( (t[:, :, mon] - mu_t) / sig_t) +
                                                             np.square( (p[:, :, mon] - mu_p)/sig_p)
                                                             - 2*rho*(t[:, :, mon]-mu_t)*(p[:, :, mon] - mu_p)
                                                             / (sig_t*sig_p)))
        return dy

    @staticmethod
    def gauss2d_resp(norm, t, mu_t, sig_t, p, mu_p, sig_p, rho):

        dy = norm * np.exp(-(1/(2 - 2*np.square(rho))) * (np.square( (t - mu_t) / sig_t) + np.square( (p - mu_p)/sig_p)
                                                          - 2*rho*(t-mu_t)*(p - mu_p) / (sig_t*sig_p)))
        return dy

    def get_day_keys(self):

        # Define the number of days in a month
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_in_month_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_of_the_year_notleap = np.cumsum(days_in_month)
        day_of_the_year_leap = np.cumsum(days_in_month_leap)

        # Initialise the dictionary to hold keys
        sow_dict = {}

        # Loop over regions
        for regind, (lat, long) in enumerate(zip(self.reg_lats, self.reg_longs)):

            # Initialise this regions entry
            sow_dict.setdefault(str(lat) + '.' + str(long), {})

            # Loop over yield collection years
            for yrind, yield_yr in enumerate(self.reg_yrs[regind, :]):

                # Extract this years sowing date and ripening time in days
                sow_month = self.sow_months[regind, yrind]
                ripe_time = int(self.ripe_days[regind, yrind])
                nmonths = ripe_time % 12

                # Ensure the correct year is used for sowing and yield collection
                if int(sow_month) + nmonths <= 12 or nmonths == 12 and int(sow_month == 1):
                    key_year = int(yield_yr)
                else:
                    key_year = int(yield_yr) - 1

                if calendar.isleap(key_year):
                    start_day = day_of_the_year_leap[int(sow_month)]
                else:
                    start_day = day_of_the_year_notleap[int(sow_month)]

                # Initialise this region's dictionary entry
                hdf_keys = np.empty(ripe_time, dtype=object)

                # Loop over months between sowing and ripening
                for nday in range(ripe_time):

                    # Compute the correct month number for this month
                    key_day = start_day + nday

                    # If month is greater than 12 this indicates the year has increased and the month number
                    # should be decremented by 12. If this results in 1 then a new year has started and the year
                    # pointer should be incremented
                    if calendar.isleap(key_year):
                        key_day %= 367
                        if key_day == 0:
                            key_day = 1
                            key_year += 1
                    else:
                        key_day %= 366
                        if key_day == 0:
                            key_day = 1
                            start_day = 1
                            key_year += 1

                    # Append this key to the dictionary under this region and year
                    # *** NOTE: these are assigned in numerical month order not chronological
                    # to aid anomaly calculation ***
                    hdf_keys[nday] = str(key_year) + '.%04d' % key_day

                # Assign keys to dictionary
                sow_dict[str(lat) + '.' + str(long)][str(yield_yr)] = hdf_keys

        return sow_dict

    def get_weather_data(self, weather):

        hdf = h5py.File('../data/SimFarm2030_' + weather + '.hdf5', 'r')

        # Loop over regions
        wthr = np.full((len(self.reg_lats), self.reg_yrs.shape[1], 366), -999)
        for llind, (lat, long) in enumerate(zip(self.reg_lats, self.reg_longs)):

            # Loop over years for each region
            for yrind, year in enumerate(self.reg_yrs[llind, :]):

                hdf_keys = self.reg_keys[str(lat) + '.' + str(long)][str(year)]

                # Initialise arrays to hold results
                for key in hdf_keys:

                    year, month = key.split('.')

                    print('Processing ', weather, str(month) + '/' + str(year) + '...', end="\r")

                    # If year is within list of years extract the relevant data
                    wthr[llind, yrind, int(month) - 1] = self.extract_region(hdf['Latitude_grid'][...],
                                                                             hdf['Longitude_grid'][...], lat, long,
                                                                             hdf[key]['daily_grid'][...],
                                                                             self.tol)

        hdf.close()

        # Assign weather data to variable
        self.wthr_dict[weather] = wthr

    def get_weather_anomaly(self, weather):

        # Get the region's weather data
        if weather not in self.wthr_dict:
            self.get_weather_data(weather)

        hdf = h5py.File('../data/SimFarm2030_' + weather + '.hdf5', 'r')

        # Get the mean weather data for each month of the year
        uk_monthly_mean = hdf['all_years_mean'][...]

        hdf.close()

        # Calculate the anomaly
        anom = self.wthr_dict[weather] - uk_monthly_mean
        self.wthr_anom_dict[weather] = anom

        return anom

    @staticmethod
    def Rolling_Median(One_D_Array, n=5):

        # Convert input array to pandas series to utilise pandas functionality
        s = pd.Series(One_D_Array)

        # Get the rolling median for a window of n size
        rolling = s.rolling(n, center=True).median()

        return rolling

    @staticmethod
    def FTP_download(ftppath, destpath, user='abowell', passwd='SimFarm2030', only_2000=True):

        # Log in to FTP server
        try:
            ftp = FTP('ftp1.ceda.ac.uk')
            ftp.login(user=user, passwd=passwd)
            print('Logged in')
        except error_perm:
            raise Exception('Log in credentials invalid')

        # Define current working directory in the FTP server
        try:
            ftp.cwd(ftppath)
        except error_perm:
            raise Exception('FTP server directory invalid')

        # Adds all files found that match critera and adds to list
        try:
            files = ftp.nlst()
        except (error_perm, resp):
            if str(resp) == '550 No files found':
                raise Exception('No files in this directory')
            else:
                raise Exception('FTP server unresponsive')

        # If only year beyond 2000 are required limit the download to these files
        if only_2000:

            # Loop over files extracting only files for years of the form 20XX
            otherfiles = []
            for name in files:
                # If file is in range include it in the files list
                if fnmatch.fnmatch(name, "*hadukgrid_uk_1km_day_20*nc"):
                    otherfiles.append(name)
        else:
            otherfiles = files

        #  Make sure destination exists
        if not os.path.exists(destpath):
            print('Destination path created at', destpath)
            os.mkdir(destpath)

        # Loops through all files in the list and downloads them to wherever specified
        for counter, item in enumerate(otherfiles):
            # Log in to FTP server once again to get current files
            ftp = FTP('ftp1.ceda.ac.uk')
            ftp.login(user=user, passwd=passwd)

            # Redefine current working directory in the FTP server
            ftp.cwd(ftppath)

            # Create file to write the data out to
            localfile = open(destpath + item, 'wb')
            ftp.retrbinary('RETR ' + item, localfile.write, 1024)
            print('File ', item, 'is downloaded. \n', counter, ' out of ', len(otherfiles))
            localfile.close()

    def train_model(self, chains=5, iter=1000, verbose=True, control={'max_treedepth': 13}):

        # Sample the model, training it with training data
        self.fit = self.model.sampling(data=self.modeldata, chains=chains, iter=iter, verbose=verbose, control=control)

        # Assign samples to object
        self.samples = self.fit.extract()
        
        # Extract the mean parameter values
        self.mean_params['mu_t'] = np.median(self.samples['mu_t'])
        self.mean_params['sig_t'] = np.median(self.samples['sigma_t'])
        self.mean_params['mu_p'] = np.median(self.samples['mu_p'])
        self.mean_params['sig_p'] = np.median(self.samples['sigma_p'])
        self.mean_params['norm'] = np.median(self.samples['norm'])
        self.mean_params['rho'] = np.median(self.samples['rho'])

    def region_predict(self, chains=5, iter=1000, verbose=True, control={'max_treedepth': 13}):

        pstart = time.time()

        self.reg_ysig = np.zeros((self.reg_lats.size, self.yield_data.shape[1]))

        for regind in range(0, self.reg_lats.size):

            boolarr = np.full_like(self.reg_lats, True, dtype=bool)
            boolarr[regind] = False

            predictdata = {'n_regions': len(self.reg_lats[boolarr]), 'n_years': self.reg_yrs.shape[1],
                           'd_temp': self.temp_anom[boolarr, :], 'd_precip': self.precip_anom[boolarr, :],
                           'd_yields': self.yield_data[boolarr, :],
                           'n_gf': self.n_gf, 'temp': np.linspace(-10, 40, self.n_gf),
                           'precip': np.linspace(0, 250, self.n_gf)}

            pfit = self.model.sampling(data=predictdata, chains=chains, iter=iter, verbose=verbose, control=control)

            samples = pfit.extract()

            # Extract the mean parameter values
            mean_params = {}
            mean_params['mu_t'] = np.median(samples['mu_t'])
            mean_params['sig_t'] = np.median(samples['sigma_t'])
            mean_params['mu_p'] = np.median(samples['mu_p'])
            mean_params['sig_p'] = np.median(samples['sigma_p'])
            mean_params['norm'] = np.median(samples['norm'])
            mean_params['rho'] = np.median(samples['rho'])

            for yrind, yr in enumerate(range(0, self.yield_data.shape[1])):
                self.reg_pred[regind, yrind] = self.gauss2d(mean_params['norm'],
                                                            self.wthr_anom_dict[self.weather[0]][regind, yrind, :],
                                                            mean_params['mu_t'], mean_params['sig_t'],
                                                            self.wthr_anom_dict[self.weather[1]][regind, yrind, :],
                                                            mean_params['mu_p'], mean_params['sig_p'],
                                                            mean_params['rho'])

                reg_yall = np.zeros(samples['mu_t'].size)

                # Loop over posteriors
                for ind in range(samples['mu_t'].size):
                    print('Full posterior... %.2f' % (ind / samples['mu_t'].size * 100), '%', end='\r')

                    # Compute temperature response
                    reg_yall[ind] = self.gauss2d(samples['norm'][ind],
                                                 self.wthr_anom_dict[self.weather[0]][regind, yrind, :],
                                                 samples['mu_t'][ind], samples['sigma_t'][ind],
                                                 self.wthr_anom_dict[self.weather[1]][regind, yrind, :],
                                                 samples['mu_p'][ind], samples['sigma_p'][ind], samples['rho'][ind])

                # Assign posteriors result removing extreme outliers where model has failed
                self.reg_ysig[regind, yrind] = np.std(reg_yall[np.where(reg_yall < 10**2)])

        print('Mean Residual', (self.yield_data - self.reg_pred).mean())
        print('Median Residual', np.median(self.yield_data - self.reg_pred))

        # Calculate residuals
        self.resi = self.yield_data - self.reg_pred

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.axhline(0.0, linestyle='--', color='k')
        ax.axhline(np.mean(self.resi), linestyle='-', color='g', label='Mean')
        ax.axhline(np.median(self.resi), linestyle='-', color='r', label='Median')

        # Plot residuals
        xshift = 0
        for num, (resi_lst, err_lst) in enumerate(zip(self.resi, self.reg_ysig)):

            ax.errorbar(np.linspace(1 + xshift, len(resi_lst) + xshift, len(resi_lst)), resi_lst, yerr=err_lst,
                        marker='+', linestyle='none', capsize=5, markersize=10)
            xshift += len(resi_lst)

        reg_mean_resi = np.mean(self.resi, axis=1)

        ax.plot(np.linspace(1, self.resi.shape[0] * self.resi.shape[1] + 1, self.resi.shape[0]),
                reg_mean_resi, label='Region Mean')

        # Label axes
        ax.set_xlabel(r'Years')
        ax.set_ylabel('$Y_{\mathrm{true}}-Y_{\mathrm{pred}}$')  # + self.metric + ' (' + self.metric_units + ')')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        fig.savefig('../model_performance/region_residuals_' + self.metric + '.png', dpi=300, bbox_inches='tight')

        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot residuals
        xshift = 0
        for num, (resi_lst, y, err_lst) in enumerate(zip(self.yield_data - self.reg_pred,
                                                         self.yield_data, self.reg_ysig)):
            ax.errorbar(np.linspace(1 + xshift, len(resi_lst) + xshift, len(resi_lst)), resi_lst / y * 100,
                        yerr=err_lst / y * 100, marker='+', linestyle='none', capsize=5, markersize=10)
            xshift += len(resi_lst)

        ax.plot(np.linspace(1, self.resi.shape[0] * self.resi.shape[1] + 1, self.resi.shape[0]),
                np.mean(self.resi / self.yield_data * 100, axis=1), label='Site Mean')

        ax.axhline(0.0, linestyle='--', color='k')
        ax.axhline(np.mean(self.resi / self.yield_data * 100), linestyle='-', color='g',
                   label='Mean (all sites/years)')
        ax.axhline(np.median(self.resi / self.yield_data * 100), linestyle='-', color='r',
                   label='Median (all sites/years)')

        # Label axes
        ax.set_xlabel(r'Years')
        ax.set_ylabel('$Y_{\mathrm{true}}-Y_{\mathrm{pred}}/Y_{\mathrm{true}}$ (%)')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        fig.savefig('../model_performance/region_pcent_residuals_' + self.metric + '.png', dpi=300, bbox_inches='tight')

        plt.close(fig)

        # df = pd.DataFrame(columns=['Region', 'Prediction'])
        # df['Region'] = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
        #                 9, 9, 9, 9, 10 , 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14,
        #                 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19,
        #                 20, 20, 20, 20, 21, 21, 21, 21]
        # df['Prediction'] = self.resi.flatten()
        #
        # fig, ax = plt.subplots(figsize=(12, 4))
        # sns.violinplot(ax=ax, x='Region', y='Prediction', data=df)
        #
        # plt.savefig('../region_violin_' + self.metric + '.png', dpi=300, bbox_inches='tight')

        print('Region Prediction', time.time() - pstart)

    def country_predict(self, year, tmod, pmod, mutmod, mupmod, cultivar):

        # Open file
        hdf = h5py.File('../data/SimFarm2030.hdf5',
                        'r+')

        # Extract latitude grid
        lat = hdf['Latitude_grid'][...]

        # List all months
        allmonths = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

        # Loop over years for each region
        # Initialise arrays to hold results
        temps = np.zeros((lat.shape[0], lat.shape[1], 12))
        rains = np.zeros((lat.shape[0], lat.shape[1], 12))
        year = str(round(year))

        # Loop over years for each region
        for mthind, month in enumerate(allmonths):

            key = str(year) + '.' + str(month)
            temps[:, :, mthind] = (hdf[key]['Monthly_mean_temperature_grid'][...]
                                   - hdf['UK_monthly_all_years_mean_temperature'][...][mthind])
            rains[:, :, mthind] = (hdf[key]['Monthly_mean_rainfall_grid'][...]
                                   - hdf['UK_monthly_all_years_mean_rainfall'][...][mthind])

        preds = self.gauss2d_country(self.mean_params['norm'], temps + tmod, self.mean_params['mu_t'] + mutmod,
                                     self.mean_params['sig_t'], rains + pmod, self.mean_params['mu_p'] + mupmod,
                                     self.mean_params['sig_p'], self.mean_params['rho'], dy=np.zeros_like(lat))
        hdf.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.imshow(np.flip(preds, axis=0), vmin=0, vmax=16)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.text(0.1, 0.9, str(year) + ': ' + cultivar, verticalalignment='center', horizontalalignment='left',
                transform=ax.transAxes, fontsize=7, color='w', bbox=dict(facecolor='k', edgecolor='w',
                                                                         boxstyle='round', alpha=0.4))

        cbar = fig.colorbar(cax)
        cbar.ax.set_ylabel(self.metric + ' (' + self.metric_units + ')')

        fig.savefig('../country_predictions/prediction_country_map_' + cultivar + '_' + str(year) + '_'
                    + str(round(mutmod, 3)) + '_'
                    + str(round(mupmod, 3)) + '_'
                    + str(round(tmod, 3)) + '_'
                    + str(round(pmod, 3)) + '.png', dpi=300, bbox_inches='tight')
        fig.clf()

        return np.flip(preds, axis=0)

    def plot_response(self):

        # Create arrays to evaluate response function at
        eval_t = np.linspace(-25 + self.mean_params['mu_t'], self.mean_params['mu_t'] + 25, 1000)
        eval_p = np.linspace(-200 + self.mean_params['mu_p'], self.mean_params['mu_p'] + 200, 1000)

        # Get grid of values
        tt, pp = np.meshgrid(eval_t, eval_p)

        # Compute temperature response
        t_resp = self.gauss2d_resp(self.mean_params['norm'], eval_t, self.mean_params['mu_t'],
                                   self.mean_params['sig_t'], 0, self.mean_params['mu_p'],
                                   self.mean_params['sig_p'], self.mean_params['rho'])

        # Compute precipitation response
        p_resp = self.gauss2d_resp(self.mean_params['norm'], 0, self.mean_params['mu_t'],
                                   self.mean_params['sig_t'], eval_p, self.mean_params['mu_p'],
                                   self.mean_params['sig_p'], self.mean_params['rho'])

        # Compute the response grid
        resp_grid = self.gauss2d_resp(self.mean_params['norm'], tt, self.mean_params['mu_t'],
                                      self.mean_params['sig_t'], pp, self.mean_params['mu_p'],
                                      self.mean_params['sig_p'], self.mean_params['rho'])

        # Set up figure
        fig = plt.figure(figsize=(9, 9.6))
        gs = gridspec.GridSpec(3, 2)
        # gs.update(wspace=0.5, hspace=0.0)
        ax1 = fig.add_subplot(gs[:2, :])
        ax2 = fig.add_subplot(gs[2, 0])
        ax3 = fig.add_subplot(gs[2, 1])

        # Plot the response functions
        cba = ax1.pcolormesh(eval_t, eval_p, resp_grid)

        cbar = fig.colorbar(cba, ax=ax1)
        cbar.ax.set_ylabel(self.metric + ' (' + self.metric_units + 'month$^{-1}$)')

        ax2.plot(eval_t, t_resp)
        ax3.plot(eval_p, p_resp)

        # Label axes
        ax1.set_xlabel(r'$\Delta T$ ($^\circ$C)')
        ax1.set_ylabel(r'$\Delta P$ (mm)')
        ax2.set_xlabel(r'$\Delta T$ ($^\circ$C)')
        ax2.set_ylabel(self.metric + ' (' + self.metric_units + 'month$^{-1}$)')
        ax3.set_xlabel(r'$\Delta P$ (mm)')
        ax3.set_ylabel(self.metric + ' (' + self.metric_units + 'month$^{-1}$)')

        # Save the figure
        fig.savefig('../model_performance/responsecurves_' + self.metric + '.png', dpi=300, bbox_inches='tight')

        return eval_t, eval_p, t_resp, p_resp, resp_grid

    def post_prior_comp(self):

        # Define prior pandas table
        df = pd.DataFrame(np.vstack((np.random.normal(1, 3, 2000), np.random.normal(0.0, 7, 2000),
                                     np.random.normal(3, 3, 2000), np.random.normal(0.0, 50, 2000),
                                     np.random.normal(25, 10, 2000), np.random.normal(0.0, 1, 2000))).T,
                          columns=[r'norm', r'$\mu_t$', r'$\sigma_t$', r'$\mu_p$', r'$\sigma_p$', r'$\rho$'])

        # Plot prior
        g = sns.PairGrid(data=df, size=2.5, diag_sharey=False)
        g.map_diag(plt.hist, color='Red', alpha=0.5)
        g.map_lower(sns.kdeplot, cmap="Reds", alpha=0.8, n_levels=10, normed=True, shade=True, shade_lowest=False)

        # Define posterior pandas table
        df = pd.DataFrame(np.vstack((self.samples['norm'], self.samples['mu_t'], self.samples['sigma_t'],
                                     self.samples['mu_p'], self.samples['sigma_p'], self.samples['rho'])).T,
                          columns=[r'norm', r'$\mu_t$', r'$\sigma_t$', r'$\mu_p$', r'$\sigma_p$', r'$\rho$'])

        # Plot posterior
        g.data = df
        g.map_diag(plt.hist, color='Blue', alpha=0.5)
        g.map_lower(sns.kdeplot, cmap="Blues", alpha=0.8, n_levels=10, normed=True, shade=True, shade_lowest=False)

        for i in range(0, 6):
            for j in range(0, 6):
                if j <= i:
                    continue
                g.axes[i, j].set_axis_off()

        # Save figure
        plt.savefig('../model_performance/posteriorPriorComp_' + self.metric + '.png', dpi=300, bbox_inches='tight')

    def country_animate(self, yrmin=1900, yrmax=2018):

        # Open file
        hdf = h5py.File('../data/SimFarm2030.hdf5',
                        'r+')

        lat = hdf['Latitude_grid'][...]

        allmonths = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

        allpred = np.zeros((lat.shape[0], lat.shape[1], len(range(yrmin, yrmax))))
        cntrywide_pred = np.zeros(len(range(yrmin, yrmax)))
        yrs = np.zeros(len(range(yrmin, yrmax)))
        for ind, year in enumerate(range(yrmin, yrmax)):

            # Loop over years for each region
            temps = np.zeros((lat.shape[0], lat.shape[1], 12))
            rains = np.zeros((lat.shape[0], lat.shape[1], 12))
            year = str(year)
            # Initialise arrays to hold results
            for mthind, month in enumerate(allmonths):
                key = str(year) + '.' + str(month)
                temps[:, :, mthind] = hdf[key]['Monthly_mean_temperature_grid'][...]
                rains[:, :, mthind] = hdf[key]['Monthly_mean_rainfall_grid'][...]

            preds = self.gauss2d_country(self.mean_params['norm'], temps, self.mean_params['mu_t'],
                                         self.mean_params['sig_t'], rains, self.mean_params['mu_p'],
                                         self.mean_params['sig_p'], self.mean_params['rho'],
                                         dy=np.zeros_like(lat))
            allpred[:, :, ind] = preds
            cntrywide_pred[ind] = preds[np.where(preds > 0)].mean()
            yrs[ind] = int(year)
        print('Country Prediction', time.time() - pstart)

        allpred -= allpred[np.where(allpred > 0)].mean()
        miny = allpred.min()
        maxy = allpred.max()

        for ind, year in enumerate(range(yrmin, yrmax)):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            cax = ax.imshow(np.flip(allpred[:, :, ind], axis=0), vmin=miny, vmax=maxy)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            ax.text(0.9, 0.9, str(year), verticalalignment='center', horizontalalignment='right',
                    transform=ax.transAxes, fontsize=7, color='w', bbox=dict(facecolor='k', edgecolor='w',
                                                                             boxstyle='round', alpha=0.4))

            cbar = fig.colorbar(cax)
            cbar.ax.set_ylabel(self.metric + ' Anomaly ' + '(' + self.metric_units + ')')

            fig.savefig('../country_predictions/prediction_country_map_anom_' + str(year) + '.png', dpi=300,
                        bbox_inches='tight')
            fig.clf()

        hdf.close()

        os.system('convert -loop 1 -delay 50 ../country_predictions/prediction_country_map_anom_*.png '
                  'country_predictions/prediction_country_map_anom_' + self.metric + '.gif')

        # Set up figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot country wide mean
        ax.plot(yrs, cntrywide_pred, label='Country Wide Mean')

        # Plot each region over the country wide mean
        for i, (regys, regyrs) in enumerate(zip(self.yield_data, self.reg_yrs)):
            years = [int(yr) for yr in regyrs]
            ax.scatter(years, regys, marker='+', label='Region ' + str(i))

        # Label axes
        ax.set_xlabel(r'Year')
        ax.set_ylabel(self.metric + ' (' + self.metric_units + ')')

        ax.tick_params(axis='x', rotation=45)

        fig.savefig('../country_predictions/Country_wide_mean_' + self.metric + '.png', dpi=300,
                    bbox_inches='tight')

    def region_all_years(self, regs=4, yrmin=1900, yrmax=2018):

        # Open file
        hdf = h5py.File('../data/SimFarm2030.hdf5',
                        'r+')

        lat = hdf['Latitude_grid'][...]

        allmonths = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

        cntrywide_pred = np.zeros(len(range(yrmin, yrmax)))
        cntrywide_max = np.zeros(len(range(yrmin, yrmax)))
        cntrywide_80 = np.zeros(len(range(yrmin, yrmax)))
        cntrywide_90 = np.zeros(len(range(yrmin, yrmax)))
        cntrywide_95 = np.zeros(len(range(yrmin, yrmax)))
        yrs = np.zeros(len(range(yrmin, yrmax)))
        for ind, year in enumerate(range(yrmin, yrmax)):

            # Loop over years for each region
            temps = np.zeros((lat.shape[0], lat.shape[1], 12))
            rains = np.zeros((lat.shape[0], lat.shape[1], 12))
            year = str(year)
            # Initialise arrays to hold results
            for mthind, month in enumerate(allmonths):
                key = str(year) + '.' + str(month)
                temps[:, :, mthind] = hdf[key]['Monthly_mean_temperature_grid'][...]
                rains[:, :, mthind] = hdf[key]['Monthly_mean_rainfall_grid'][...]

            preds = self.gauss2d_country(self.mean_params['norm'], temps, self.mean_params['mu_t'],
                                            self.mean_params['sig_t'], rains, self.mean_params['mu_p'],
                                            self.mean_params['sig_p'], self.mean_params['rho'],
                                            dy=np.zeros_like(lat))

            cntrywide_pred[ind] = np.median(preds[np.where(preds > 0)])
            cntrywide_max[ind] = np.max(preds[np.where(preds > 0)])
            cntrywide_80[ind] = np.percentile(preds[np.where(preds > 0)], 80)
            cntrywide_90[ind] = np.percentile(preds[np.where(preds > 0)], 90)
            cntrywide_95[ind] = np.percentile(preds[np.where(preds > 0)], 95)
            yrs[ind] = int(year)
        print('Country Prediction', time.time() - pstart)

        # Set up figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Loop over region
        for regind in range(0, regs):

            # Loop over year
            reg_pred_yrs = np.zeros(len(range(yrmin, yrmax)))
            for yrind, yr in enumerate(range(yrmin, yrmax)):

                print('Proceessing Region', regind, yr, '...', end="\r")

                # Loop over months
                temps = np.zeros(len(allmonths))
                rains = np.zeros(len(allmonths))
                for mnthind, mnth in enumerate(allmonths):

                    temps[mnthind] = self.extract_region(hdf['Latitude_grid'][...], hdf['Longitude_grid'][...],
                                                         self.reg_lats[regind], self.reg_longs[regind],
                                                         hdf[str(yr) + '.' + mnth]['Monthly_mean_temperature_grid'][...],
                                                         self.tol) \
                                     - hdf['UK_monthly_all_years_mean_temperature'][...][mnthind]
                    rains[mnthind] = self.extract_region(hdf['Latitude_grid'][...], hdf['Longitude_grid'][...],
                                                         self.reg_lats[regind], self.reg_longs[regind],
                                                         hdf[str(yr) + '.' + mnth]['Monthly_mean_rainfall_grid'][...],
                                                         self.tol) \
                                     - hdf['UK_monthly_all_years_mean_rainfall'][...][mnthind]
                reg_pred_yrs[yrind] = self.gauss2d(self.mean_params['norm'], temps, self.mean_params['mu_t'],
                                                   self.mean_params['sig_t'], rains, self.mean_params['mu_p'],
                                                   self.mean_params['sig_p'], self.mean_params['rho'])
            # Plot this region
            ax.plot(yrs, reg_pred_yrs, linestyle='--')

        hdf.close()

        # # Plot country wide median and max
        # ax.plot(yrs, cntrywide_pred, label='50th Percentile')
        # ax.plot(yrs, cntrywide_max, label='Maximum')
        # ax.plot(yrs, cntrywide_80, label='80th Percentile')
        # ax.plot(yrs, cntrywide_90, label='90th Percentile')
        # ax.plot(yrs, cntrywide_95, label='95th Percentile')

        # Plot each region over the country wide mean
        for i, (regys, regyrs) in enumerate(zip(self.yield_data, self.reg_yrs)):
            years = [int(yr) for yr in regyrs]
            ax.scatter(years, regys, marker='+')

        # Label axes
        ax.set_xlabel(r'Year')
        ax.set_ylabel(self.metric + ' (' + self.metric_units + ')')
        ax.tick_params(axis='x', rotation=45)

        # Draw legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        fig.savefig('../country_predictions/Country_wide_mean_region_comp_' + self.metric + '.png', dpi=300,
                    bbox_inches='tight')


if __name__ == '__main__':

    # ftp_paths = ['/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/tas/mon/v20181126',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/sun/mon/v20181126',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/mon/v20181126']

    # ftp_paths = ['/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/day/v20181126',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/tasmax/day/v20181126',
    #              '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/tasmin/day/v20181126']

    # ftp_paths = ['/badc/ukcp18/data/land-rcm/uk/12km/rcp85/01/tas/mon/latest',
    #              '/badc/ukcp18/data/land-rcm/uk/12km/rcp85/01/pr/mon/latest']
    #
    # destpaths = ['data/Rain/',
    #              '/Volumes/My Passport/SimFarm_daily/Temp_max/',
    #              '/Volumes/My Passport/SimFarm_daily/Temp_min']
    #
    # for path, destpath in zip(ftp_paths, destpaths):
    #     cultivarModel.FTP_download(path, destpath)

    start = time.time()
    # Define data to train the model
    yields = np.loadtxt('../example_data/Yields.txt').T
    regions = np.loadtxt('../example_data/Regions.txt')
    region_lats = regions[:, 0]
    region_longs = regions[:, 1]
    years = np.loadtxt('../example_data/Years.txt',
                       dtype=str).T
    sow_month = np.loadtxt('../example_data/sowmonth.txt').T
    ripe_time = np.loadtxt('../example_data/ripedays.txt').T
    ripe_time[np.where(ripe_time == -999)] = np.mean(ripe_time[np.where(ripe_time != -999)])

    # yields_med = np.median(yields, axis=1)

    # yield1 = np.zeros((21, 4))
    # for i, (ys, med) in enumerate(zip(yields, yields_med)):
    #     for j, y in enumerate(ys):
    #         yield1[i, j] = y - med
    #
    # yields = yield1
    #
    gm = pystan.StanModel(file='../Stan_models/2d-gaussian_with_correlation_anom_daily.stan')

    print('Model', time.time() - start)
    tstart = time.time()
    simfarm = cultivarModel(region_lats, region_longs, years, 'Solstice', sow_month, ripe_time, gm, yields,
                            region_tol=0.25, n_gf=40, weather=['temperature', 'rainfall'], metric='Yield',
                            metric_units='t Ha$^{-1}$')
    simfarm.train_model(chains=5, iter=1000, verbose=True, control={'max_treedepth': 13})
    print('Train', time.time() - tstart)

    pstart = time.time()

    # simfarm.region_all_years()
    # simfarm.country_animate()
    simfarm.post_prior_comp()
    simfarm.plot_response()

    # Write out object as pickle
    with open('../cultivar_models/' + simfarm.cult + '_' + simfarm.metric + '_modeltestdaily.pck', 'wb') as pfile1:
        pickle.dump(simfarm, pfile1)

    simfarm.region_predict()

    # # Write out object as pickle
    # with open('cultivar_models/Solstice_Yield_modeltestdaily.pck', 'rb') as pfile1:
    #     mod = pickle.load(pfile1)

