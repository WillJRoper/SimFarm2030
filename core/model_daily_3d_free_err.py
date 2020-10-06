import numpy as np
import scipy.stats as stat
import h5py
import corner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utilities
import os
from multiprocessing import Pool
import emcee
import seaborn as sns
import warnings
import time
import pandas as pd
import datetime
from calendar import monthrange


warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

sns.set_style("whitegrid")
sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12})


class cultivarModel:

    def __init__(self, cultivar, region_tol=0.25, weather=("temperature", "rainfall", "sunshine"), metric="yield",
                 metric_units="t/Ha", extract_flag=False, initial_guess=(10, 3375, 380, 705, 220, 1500, 150, 0, 0, 0, 0),
                 initial_spread=(15, 500, 300, 400, 300, 600, 300, 1, 1, 1, 10)):

        start = time.time()

        # Get the inputs
        data = utilities.extract_data("../example_data/" + cultivar + "_Data.csv")
        region_lats, region_longs, years, ripe_days, yields, sow_day, sow_month = data

        self.reg_lats = region_lats
        self.reg_longs = region_longs
        self.reg_yrs = years
        self.sow_days = sow_day
        self.sow_months = sow_month
        self.sow_year = years - 1
        self.ripe_days = ripe_days
        self.reg_keys = self.get_day_keys()
        self.reg_mth_keys = self.get_month_keys()
        self.yield_data = yields
        self.cult = cultivar
        self.tol = region_tol
        self.wthr_dict = {}
        self.wthr_anom_dict = {}
        self.mean_params = {}
        self.fit = None
        self.extract_flag = extract_flag

        self.initial_guess = initial_guess
        self.initial_spread = initial_spread
        self.samples = {}
        self.reg_pred = np.zeros_like(self.yield_data)
        self.resi = None
        self.weather = weather
        self.metric = metric
        self.metric_units = metric_units

        # Open file
        try:

            hdf = h5py.File("../Climate_Data/Region_Climate_" + self.cult + ".hdf5", "r")

            self.temp_anom, self.temp = hdf["Temperature_Anomaly"][...], hdf["Temperature"][...]
            self.temp[self.temp <= 0] = np.nan
            print("Temperature Extracted")
            self.precip_anom, self.precip = hdf["Rainfall_Anomaly"][...], hdf["Rainfall"][...]
            print("Rainfall Extracted")
            self.sun_anom, self.sun = hdf["Sunshine_Anomaly"][...], hdf["Sunshine"][...]
            print("Sunshine Extracted")

            hdf.close()

        except KeyError:
            extract_flag = True
        except OSError:
            extract_flag = True

        if extract_flag:

            self.temp_anom, self.temp = self.get_weather_anomaly(weather[0])
            self.temp[self.temp <= 0] = np.nan
            print("Temperature Extracted")
            self.precip_anom, self.precip = self.get_weather_anomaly(weather[1])
            print("Rainfall Extracted")
            self.sun_anom, self.sun = self.get_weather_anomaly_monthly(weather[2])
            print("Sunshine Extracted")

            hdf = h5py.File("../Climate_Data/Region_Climate_" + self.cult + ".hdf5", "w")

            hdf.create_dataset("Temperature", shape=self.temp.shape, dtype=self.temp.dtype,
                               data=self.temp, compression="gzip")
            hdf.create_dataset("Temperature_Anomaly", shape=self.temp_anom.shape, dtype=self.temp_anom.dtype,
                               data=self.temp_anom, compression="gzip")
            hdf.create_dataset("Rainfall", shape=self.precip.shape, dtype=self.precip.dtype,
                               data=self.precip, compression="gzip")
            hdf.create_dataset("Rainfall_Anomaly", shape=self.precip_anom.shape, dtype=self.precip_anom.dtype,
                               data=self.precip_anom, compression="gzip")
            hdf.create_dataset("Sunshine", shape=self.sun.shape, dtype=self.sun.dtype,
                               data=self.sun, compression="gzip")
            hdf.create_dataset("Sunshine_Anomaly", shape=self.sun_anom.shape, dtype=self.sun_anom.dtype,
                               data=self.sun_anom, compression="gzip")

            hdf.close()

        # Compute thermal days and total rainfall
        self.therm_days = np.nansum(self.temp, axis=1)
        self.tot_precip = np.nansum(self.precip, axis=1)
        self.tot_sun = np.nansum(self.sun, axis=1)

        print("Input extracted:", time.time() - start)

    @staticmethod
    def extract_region(lat, long, region_lat, region_long, weather, tol):

        # Get the boolean array for points within tolerence
        bool_cond = np.logical_and(np.abs(lat - region_lat) < tol, np.abs(long - region_long) < tol)

        # Get the extracted region
        ex_reg = weather[bool_cond]

        # Remove any nan and set them to 0 these correspond to ocean
        ex_reg = ex_reg[ex_reg < 1e8]

        if ex_reg.size == 0:
            print("Region not in coords:", region_lat, region_long)
            return np.nan
        else:
            return np.mean(ex_reg)

    @staticmethod
    def gauss2d_country(norm, t, mu_t, sig_t, p, mu_p, sig_p, rho, dy=0):

        for mon in range(0, 12):
            dy += norm * np.exp(-(1 / (2 - 2 * np.square(rho))) * (np.square((t[:, :, mon] - mu_t) / sig_t) +
                                                                   np.square((p[:, :, mon] - mu_p) / sig_p)
                                                                   - 2 * rho * (t[:, :, mon] - mu_t) * (
                                                                               p[:, :, mon] - mu_p)
                                                                   / (sig_t * sig_p)))
        return dy

    @staticmethod
    def gauss2d_resp(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho):

        t_term = ((t - mu_t) / sig_t) ** 2
        p_term = ((p - mu_p) / sig_p) ** 2
        tp_term = 2 * rho * (t - mu_t) * (p - mu_p) / (sig_t * sig_p)

        dy = norm * np.exp(-(1 / (2 - 2 * rho * rho)) * (t_term + p_term - tp_term))

        return dy

    def get_day_keys(self):

        # Initialise the dictionary to hold keys
        sow_dict = {}

        # Loop over regions
        for regind, (lat, long, sow_yr) in enumerate(zip(self.reg_lats, self.reg_longs, self.sow_year)):

            # Initialise this regions entry
            sow_dict.setdefault(str(lat) + "." + str(long), {})

            # Extract this years sowing date and ripening time in days
            sow_day = self.sow_days[regind]
            sow_month = self.sow_months[regind]
            ripe_time = self.ripe_days[regind]
            sow_date = datetime.date(year=sow_yr, month=int(sow_month), day=int(sow_day))

            # Initialise this region"s dictionary entry
            hdf_keys = np.empty(ripe_time + 1, dtype=object)

            # Loop over months between sowing and ripening
            for nday in range(ripe_time + 1):

                # Compute the correct month number for this month
                key_date = sow_date + datetime.timedelta(days=nday)

                # Append this key to the dictionary under this region in chronological order
                hdf_keys[nday] = str(key_date.year) + "_%03d" % key_date.month + "_%04d" % key_date.day

            # Assign keys to dictionary
            sow_dict[str(lat) + "." + str(long)][str(sow_yr)] = hdf_keys

        return sow_dict

    def get_month_keys(self):

        # Initialise the dictionary to hold keys
        sow_dict = {}

        # Loop over regions
        for regind, (lat, long, sow_yr) in enumerate(zip(self.reg_lats, self.reg_longs, self.sow_year)):

            # Initialise this regions entry
            sow_dict.setdefault(str(lat) + "." + str(long), {})

            # Extract this years sowing date and ripening time in days
            sow_day = self.sow_days[regind]
            sow_month = self.sow_months[regind]
            ripe_time = self.ripe_days[regind]
            sow_date = datetime.date(year=sow_yr, month=int(sow_month), day=int(sow_day))

            # Initialise this region"s dictionary entry
            hdf_keys = []

            # Loop over months between sowing and ripening
            for ndays in range(ripe_time + 1):

                # Compute the correct month number for this month
                key_date = sow_date + datetime.timedelta(days=ndays)

                # Append this key to the dictionary under this region in chronological order
                hdf_keys.append(str(key_date.year) + "_%03d" % key_date.month)

            # Assign keys to dictionary
            sow_dict[str(lat) + "." + str(long)][str(sow_yr)] = np.unique(hdf_keys)

        return sow_dict

    def get_weather_anomaly(self, weather):

        hdf = h5py.File("../SimFarm2030_" + weather + ".hdf5", "r")

        # Get the mean weather data for each month of the year
        uk_monthly_mean = hdf["all_years_mean"][...]

        lats = hdf["Latitude_grid"][...]
        longs = hdf["Longitude_grid"][...]

        # Loop over regions
        anom = np.full((len(self.reg_lats), 400), np.nan)
        wthr = np.full((len(self.reg_lats), 400), np.nan)
        for llind, (lat, long, year) in enumerate(zip(self.reg_lats, self.reg_longs, self.sow_year)):

            hdf_keys = self.reg_keys[str(lat) + "." + str(long)][str(year)]

            # Initialise arrays to hold results
            key_ind = 0
            for key in hdf_keys:

                year, month, day = key.split("_")

                wthr_grid = hdf[key]["daily_grid"][...]

                ex_reg = self.extract_region(lats, longs, lat, long, wthr_grid, self.tol)

                # If year is within list of years extract the relevant data
                wthr[llind, key_ind] = ex_reg
                anom[llind, key_ind] = ex_reg - uk_monthly_mean[int(day) - 1]
                key_ind += 1

        # Assign weather data to variable
        self.wthr_anom_dict[weather] = anom

        hdf.close()

        return anom, wthr

    def get_weather_anomaly_monthly(self, weather):

        hdf = h5py.File("../SimFarm2030_" + weather + ".hdf5", "r")

        # Get the mean weather data for each month of the year
        uk_monthly_mean = hdf["all_years_mean"][...]

        lats = hdf["Latitude_grid"][...]
        longs = hdf["Longitude_grid"][...]

        # Loop over regions
        anom = np.full((len(self.reg_lats), 15), np.nan)
        wthr = np.full((len(self.reg_lats), 15), np.nan)
        for llind, (lat, long, year) in enumerate(zip(self.reg_lats, self.reg_longs, self.sow_year)):

            hdf_keys = self.reg_mth_keys[str(lat) + "." + str(long)][str(year)]

            # Initialise arrays to hold results
            key_ind = 0
            for key in hdf_keys:

                year, month = key.split("_")

                wthr_grid = hdf[key]["monthly_grid"][...]

                ex_reg = self.extract_region(lats, longs, lat, long, wthr_grid, self.tol)

                # If year is within list of years extract the relevant data
                wthr[llind, key_ind] = ex_reg
                anom[llind, key_ind] = ex_reg - uk_monthly_mean[int(month) - 1]
                key_ind += 1

        # Assign weather data to variable
        self.wthr_anom_dict[weather] = anom

        hdf.close()

        return anom, wthr

    @staticmethod
    def gauss2d(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho):

        t_term = ((t - mu_t) / sig_t) ** 2
        p_term = ((p - mu_p) / sig_p) ** 2
        tp_term = 2 * rho * (t - mu_t) * (p - mu_p) / (sig_t * sig_p)

        dy = norm * np.exp(-0.5 / (1 - rho * rho) * (t_term + p_term - tp_term))

        return dy

    @staticmethod
    def gauss3d(norm, t, mu_t, sig_t, p, mu_p, sig_p, s, mu_s, sig_s, rho_tp, rho_ts, rho_ps):

        dy = norm * np.exp(-(0.5 * 1 / (1 - np.square(rho_tp) - np.square(rho_ts) - np.square(rho_ps)
                                        + 2 * rho_tp * rho_ts * rho_ps))
                           * (np.square((t - mu_t) / sig_t)
                              + np.square((p - mu_p) / sig_p) + np.square((s - mu_s) / sig_s)
                              + 2 * ((t - mu_t) * (p - mu_p) * (rho_ts * rho_ps - rho_tp) / (sig_t * sig_p)
                                     + (t - mu_t) * (s - mu_s) * (rho_tp * rho_ts - rho_ps)
                                     / (sig_t * sig_s) + (p - mu_p) * (s - mu_s)
                                     * (rho_tp * rho_ts - rho_ps) / (sig_s * sig_p))))

        return dy

    def log_likelihood_3d(self, theta, t, p, s, y, yerr):

        # Extract parameter values
        norm, mu_t, sig_t, mu_p, sig_p, mu_s, sig_s, rho_tp, rho_ts, rho_ps, sig_y = theta

        # Define model
        model = self.gauss3d(norm, t, mu_t, sig_t, p, mu_p, sig_p, s, mu_s, sig_s, rho_tp, rho_ts, rho_ps)

        sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * sig_y)
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior_3d(self, theta):

        # Extract parameter values
        norm, mu_t, sig_t, mu_p, sig_p, mu_s, sig_s, rho_tp, rho_ts, rho_ps, sig_y = theta

        # The only parameters with a lower bound are norm and the sigmas
        cond = (0 < norm < 20
                and 2000 < mu_t < 4050 and 0 < sig_t < 10000
                and 350 < mu_p < 1700 and 0 < sig_p < 10000
                and 1000 < mu_s < 2000 and 0 < sig_s < 10000
                and -1 <= rho_tp <= 1 and -1 <= rho_ts <= 1 and -1 <= rho_ps <= 1
                and -50 < sig_y < 50)
        if cond:

            # # Define ln(prior) for each prior
            normpdf = stat.norm.pdf
            norm_lnprob = np.log(normpdf(norm, loc=self.initial_guess[0], scale=self.initial_spread[0]))
            mut_lnprob = np.log(normpdf(mu_t, loc=self.initial_guess[1], scale=self.initial_spread[1]))
            sigt_lnprob = np.log(normpdf(sig_t, loc=self.initial_guess[2], scale=self.initial_spread[2]))
            mup_lnprob = np.log(normpdf(mu_p, loc=self.initial_guess[3], scale=self.initial_spread[3]))
            sigp_lnprob = np.log(normpdf(sig_p, loc=self.initial_guess[4], scale=self.initial_spread[4]))
            mus_lnprob = np.log(normpdf(mu_s, loc=self.initial_guess[5], scale=self.initial_spread[5]))
            sigs_lnprob = np.log(normpdf(sig_s, loc=self.initial_guess[6], scale=self.initial_spread[6]))
            rhotp_lnprob = np.log(normpdf(rho_tp, loc=self.initial_guess[7], scale=self.initial_spread[7]))
            rhots_lnprob = np.log(normpdf(rho_ts, loc=self.initial_guess[8], scale=self.initial_spread[8]))
            rhops_lnprob = np.log(normpdf(rho_ps, loc=self.initial_guess[9], scale=self.initial_spread[9]))
            sigy_lnprob = np.log(normpdf(sig_y, loc=self.initial_guess[10], scale=self.initial_spread[10]))

            return norm_lnprob + mut_lnprob + sigt_lnprob + mup_lnprob + sigp_lnprob + mus_lnprob + sigs_lnprob \
                   + rhotp_lnprob + rhots_lnprob + rhops_lnprob + sigy_lnprob
            # return 0
        else:
            return -np.inf

    def log_probability_3d(self, theta, t, p, s, y, yerr):
        lp = self.log_prior_3d(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_3d(theta, t, p, s, y, yerr)

    def train_model(self, nsample=5000, nwalkers=500):

        temp = self.therm_days
        rain = self.tot_precip
        sun = self.tot_sun

        yields = self.yield_data
        yerr = np.std(self.yield_data)

        ndim = len(self.initial_guess)

        p0 = np.random.randn(nwalkers, ndim) * 0.0001 + self.initial_guess

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_3d,
                                            args=(temp, rain, sun, yields, yerr), pool=pool, threads=8)

            # Run 200 steps as a burn-in.
            print("Burning in ...")
            pos, prob, state = sampler.run_mcmc(p0, 200)

            # Reset the chain to remove the burn-in samples.
            sampler.reset()

            print("Running MCMC ...")
            pos, prob, state = sampler.run_mcmc(p0, nsample, progress=True, rstate0=state)

        self.model = sampler

        # Print out the mean acceptance fraction. In general, acceptance_fraction
        # has an entry for each walker so, in this case, it is a 250-dimensional
        # vector.
        af = sampler.acceptance_fraction
        print("Mean acceptance fraction:", np.mean(af))
        af_msg = '''As a rule of thumb, the acceptance fraction (af) should be 
                                    between 0.2 and 0.5
                    If af < 0.2 decrease the a parameter
                    If af > 0.5 increase the a parameter
                    '''
        
        print(af_msg)

        flat_samples = sampler.get_chain(discard=1000, thin=100, flat=True)
        self.flat_samples = flat_samples

        # Extract fitted parameters
        d = self.mean_params
        maxprob_indice = np.argmax(prob)
        self.maxprob_params = pos[maxprob_indice]
        self.fitted_params = np.median(flat_samples, axis=0)
        d["norm"], d['mu_t'], d["sig_t"], d['mu_p'], d["sig_p"], d['mu_s'], d["sig_s"], d["rho_tp"], d["rho_ts"], d["rho_ps"], d["sig_y"] = self.fitted_params

        # Extract the samples
        d = self.samples
        d["norm"], d['mu_t'], d["sig_t"], d['mu_p'], d["sig_p"], d['mu_s'], d["sig_s"], d["rho_tp"], d["rho_ts"], d["rho_ps"], d["sig_y"] = [flat_samples[:, i] for i in range(ndim)]

        # Extract the errors on the fitted parameters
        self.param_errors = np.std(flat_samples, axis=0)
        norm_err, mu_t_err, sig_t_err, mu_p_err, sig_p_err, mu_s_err, sig_s_err, rho_tp_err, rho_ts_err, rho_ps_err, sig_y_err = self.param_errors

        print("================ Model Parameters ================")
        print("norm = %.3f +/- %.3f" % (self.mean_params['norm'], norm_err))
        print("mu_t = %.3f +/- %.3f" % (self.mean_params['mu_t'], mu_t_err))
        print("sig_t = %.3f +/- %.3f" % (self.mean_params["sig_t"], sig_t_err))
        print("mu_p = %.3f +/- %.3f" % (self.mean_params['mu_p'], mu_p_err))
        print("sig_p = %.3f +/- %.3f" % (self.mean_params["sig_p"], sig_p_err))
        print("mu_s = %.3f +/- %.3f" % (self.mean_params['mu_s'], mu_s_err))
        print("sig_s = %.3f +/- %.3f" % (self.mean_params["sig_s"], sig_s_err))
        print("rho_tp = %.3f +/- %.3f" % (self.mean_params["rho_tp"], rho_tp_err))
        print("rho_ts = %.3f +/- %.3f" % (self.mean_params["rho_ts"], rho_ts_err))
        print("rho_ps = %.3f +/- %.3f" % (self.mean_params["rho_ps"], rho_ps_err))
        print("sig_y = %.3f +/- %.3f" % (self.mean_params["sig_y"], sig_y_err))

    def train_and_validate_model(self, split=0.7, nsample=5000, nwalkers=500):

        # Compute the ratio to split by
        size = self.therm_days.shape[0]
        predict_size = int(size * (1 - split))

        rand_inds = np.random.choice(np.arange(size), predict_size)
        okinds = np.zeros(size, dtype=bool)
        okinds[rand_inds] = True

        self.train_temp = self.therm_days[~okinds]
        self.train_rain = self.tot_precip[~okinds]
        self.train_sun = self.tot_sun[~okinds]
        self.train_yields = self.yield_data[~okinds]
        self.predict_temp = self.therm_days[okinds]
        self.predict_rain = self.tot_precip[okinds]
        self.predict_sun = self.tot_sun[okinds]
        self.predict_yields = self.yield_data[okinds]
        self.yerr = np.std(self.yield_data)

        self.predict_years = self.reg_yrs[okinds]
        self.predict_lat = self.reg_lats[okinds]
        self.predict_long = self.reg_longs[okinds]
        self.predict_sow_month = self.sow_months[okinds]

        ndim = len(self.initial_guess)

        p0 = np.random.randn(nwalkers, ndim) * 0.001 + self.initial_guess

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_3d,
                                        args=(self.train_temp, self.train_rain,
                                              self.train_sun, self.train_yields, self.yerr))

        # Run 200 steps as a burn-in.
        print("Burning in ...")
        pos, prob, state = sampler.run_mcmc(p0, 200)

        # Reset the chain to remove the burn-in samples.
        sampler.reset()

        print("Running MCMC ...")
        pos, prob, state = sampler.run_mcmc(p0, nsample, progress=True, rstate0=state)

        self.model = sampler

        # Print out the mean acceptance fraction. In general, acceptance_fraction
        # has an entry for each walker so, in this case, it is a 250-dimensional
        # vector.
        af = sampler.acceptance_fraction
        print("Mean acceptance fraction:", np.mean(af))
        af_msg = '''As a rule of thumb, the acceptance fraction (af) should be 
                                    between 0.2 and 0.5
                    If af < 0.2 decrease the a parameter
                    If af > 0.5 increase the a parameter
                    '''

        print(af_msg)

        flat_samples = sampler.get_chain(discard=1000, thin=100, flat=True)
        self.flat_samples = flat_samples

        # Extract fitted parameters
        d = self.mean_params
        maxprob_indice = np.argmax(prob)
        self.maxprob_params = pos[maxprob_indice]
        self.fitted_params = np.median(flat_samples, axis=0)
        d["norm"], d['mu_t'], d["sig_t"], d['mu_p'], d["sig_p"], d['mu_s'], d["sig_s"], d["rho_tp"], d["rho_ts"], d["rho_ps"], d["sig_y"] = self.fitted_params

        # Extract the samples
        d = self.samples
        d["norm"], d['mu_t'], d["sig_t"], d['mu_p'], d["sig_p"], d['mu_s'], d["sig_s"], d["rho_tp"], d["rho_ts"], d["rho_ps"], d["sig_y"] = [flat_samples[:, i] for i in range(ndim)]

        # Extract the errors on the fitted parameters
        self.param_errors = np.std(flat_samples, axis=0)
        norm_err, mu_t_err, sig_t_err, mu_p_err, sig_p_err, mu_s_err, sig_s_err, rho_tp_err, rho_ts_err, rho_ps_err, sig_y_err = self.param_errors

        print("================ Model Parameters ================")
        print("norm = %.3f +/- %.3f" % (self.mean_params['norm'], norm_err))
        print("mu_t = %.3f +/- %.3f" % (self.mean_params['mu_t'], mu_t_err))
        print("sig_t = %.3f +/- %.3f" % (self.mean_params["sig_t"], sig_t_err))
        print("mu_p = %.3f +/- %.3f" % (self.mean_params['mu_p'], mu_p_err))
        print("sig_p = %.3f +/- %.3f" % (self.mean_params["sig_p"], sig_p_err))
        print("mu_s = %.3f +/- %.3f" % (self.mean_params['mu_s'], mu_s_err))
        print("sig_s = %.3f +/- %.3f" % (self.mean_params["sig_s"], sig_s_err))
        print("rho_tp = %.3f +/- %.3f" % (self.mean_params["rho_tp"], rho_tp_err))
        print("rho_ts = %.3f +/- %.3f" % (self.mean_params["rho_ts"], rho_ts_err))
        print("rho_ps = %.3f +/- %.3f" % (self.mean_params["rho_ps"], rho_ps_err))
        print("sig_y = %.3f +/- %.3f" % (self.mean_params["sig_y"], sig_y_err))

        # Calculate the predicted results
        preds = self.gauss3d(self.mean_params['norm'], self.predict_temp, self.mean_params['mu_t'],
                             self.mean_params['sig_t'], self.predict_rain, self.mean_params['mu_p'],
                             self.mean_params['sig_p'], self.predict_sun, self.mean_params['mu_s'],
                             self.mean_params['sig_s'], self.mean_params["rho_tp"], 
                             self.mean_params["rho_ts"], self.mean_params["rho_ps"])
        print(preds)

        self.preds = preds

        # Calculate the percentage residual
        resi = (1 - (preds / self.predict_yields)) * 100
        print(resi)
        print(np.mean(resi))
        print(np.median(resi))

        self.resi = resi

        # Plot results
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(np.arange(preds.size), resi, marker="+", label="Regions")
        ax.axhline(np.mean(resi), linestyle="-", color="k", label="Mean")
        ax.axhline(np.median(resi), linestyle="--", color="k", label="Median")

        ax.set_xlabel("Region")
        ax.set_ylabel("$1 - Y_{\mathrm{Pred}} / Y_{\mathrm{True}}$ (%)")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        fig.savefig("../model_performance/Validation/" + self.cult + "_3d.png", bbox_inches="tight")

    def plot_walkers(self):

        ndim = len(self.initial_guess)

        for i in range(ndim):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            res = ax.plot(self.model.chain[:, :, i].T, "-", color="k", alpha=0.3)
            fig.savefig(f"../model_performance/Chains/samplerchain_{i}_" + self.cult + "_daily_3d.png", bbox_inches="tight")
            plt.close(fig)

    def plot_response(self):

        # Create arrays to evaluate response function at
        eval_t = np.linspace(0, 6000, 1000)
        eval_p = np.linspace(0, 3500, 1000)
        eval_s = np.linspace(0, 3500, 1000)

        # Get grid of values
        tp_tt, tp_pp = np.meshgrid(eval_t, eval_p)
        ts_tt, ts_ss = np.meshgrid(eval_t, eval_s)
        ps_pp, ps_ss = np.meshgrid(eval_p, eval_s)

        # Compute temperature response
        t_resp = self.gauss3d(self.mean_params['norm'], eval_t, self.mean_params['mu_t'],
                             self.mean_params['sig_t'], 0, self.mean_params['mu_p'],
                             self.mean_params['sig_p'], 0, self.mean_params['mu_s'],
                             self.mean_params['sig_s'], self.mean_params["rho_tp"],
                             self.mean_params["rho_ts"], self.mean_params["rho_ps"])

        # Compute precipitation response
        p_resp = self.gauss3d(self.mean_params['norm'], 0, self.mean_params['mu_t'],
                             self.mean_params['sig_t'], eval_p, self.mean_params['mu_p'],
                             self.mean_params['sig_p'], 0, self.mean_params['mu_s'],
                             self.mean_params['sig_s'], self.mean_params["rho_tp"],
                             self.mean_params["rho_ts"], self.mean_params["rho_ps"])

        # Compute sunshine response
        s_resp = self.gauss3d(self.mean_params['norm'], 0, self.mean_params['mu_t'],
                             self.mean_params['sig_t'], 0, self.mean_params['mu_p'],
                             self.mean_params['sig_p'], eval_s, self.mean_params['mu_s'],
                             self.mean_params['sig_s'], self.mean_params["rho_tp"],
                             self.mean_params["rho_ts"], self.mean_params["rho_ps"])


        # Compute the response grids
        resp_grid_tp = self.gauss2d_resp(tp_tt, self.mean_params["norm"], self.mean_params["mu_t"],
                                         self.mean_params["sig_t"], tp_pp, self.mean_params["mu_p"],
                                         self.mean_params["sig_p"], self.mean_params["rho_tp"])
        resp_grid_ts = self.gauss2d_resp(ts_tt, self.mean_params["norm"], self.mean_params["mu_t"],
                                         self.mean_params["sig_t"], ts_ss, self.mean_params["mu_s"],
                                         self.mean_params["sig_s"], self.mean_params["rho_ts"])
        resp_grid_ps = self.gauss2d_resp(ps_pp, self.mean_params["norm"], self.mean_params["mu_p"],
                                         self.mean_params["sig_p"], ps_ss, self.mean_params["mu_s"],
                                         self.mean_params["sig_s"], self.mean_params["rho_ps"])

        # Set up figure
        fig = plt.figure(figsize=(9, 12))
        gs = gridspec.GridSpec(3, 6)
        gs.update(wspace=0.4, hspace=0.3)
        ax1 = fig.add_subplot(gs[:2, :2])
        ax2 = fig.add_subplot(gs[:2, 2:4])
        ax3 = fig.add_subplot(gs[:2, 4:])
        ax4 = fig.add_subplot(gs[2, :2])
        ax5 = fig.add_subplot(gs[2, 2:4])
        ax6 = fig.add_subplot(gs[2, 4:])

        # Plot the response functions
        cba1 = ax1.pcolormesh(eval_t, eval_p, resp_grid_tp)
        cba2 = ax2.pcolormesh(eval_t, eval_s, resp_grid_ts)
        cba3 = ax3.pcolormesh(eval_p, eval_s, resp_grid_ps)

        # Add colorbars
        cax1 = ax1.inset_axes([0.05, 0.075, 0.9, 0.03])
        cax2 = ax2.inset_axes([0.05, 0.075, 0.9, 0.03])
        cax3 = ax3.inset_axes([0.05, 0.075, 0.9, 0.03])
        cbar1 = fig.colorbar(cba1, cax=cax1, orientation="horizontal")
        cbar2 = fig.colorbar(cba2, cax=cax2, orientation="horizontal")
        cbar3 = fig.colorbar(cba3, cax=cax3, orientation="horizontal")

        # Label colorbars
        cbar1.ax.set_xlabel(self.metric + " (" + self.metric_units + "month$^{-1}$)", fontsize=10, color='k', labelpad=5)
        cbar1.ax.xaxis.set_label_position('top')
        cbar1.ax.tick_params(axis='x', labelsize=10, color='k', labelcolor='k')
        cbar2.ax.set_xlabel(self.metric + " (" + self.metric_units + "month$^{-1}$)", fontsize=10, color='k', labelpad=5)
        cbar2.ax.xaxis.set_label_position('top')
        cbar2.ax.tick_params(axis='x', labelsize=10, color='k', labelcolor='k')
        cbar3.ax.set_xlabel(self.metric + " (" + self.metric_units + "month$^{-1}$)", fontsize=10, color='k', labelpad=5)
        cbar3.ax.xaxis.set_label_position('top')
        cbar3.ax.tick_params(axis='x', labelsize=10, color='k', labelcolor='k')

        ax4.plot(eval_t, t_resp)
        ax5.plot(eval_p, p_resp)
        ax6.plot(eval_s, s_resp)

        # Label axes
        ax1.set_xlabel(r"Thermal days ($^\circ$C days)")
        ax1.set_ylabel(r"$\sum P$ (mm)")
        ax2.set_xlabel(r"Thermal days ($^\circ$C days)")
        ax2.set_ylabel(r"$\sum S$ (hrs)")
        ax3.set_xlabel(r"$\sum P$ (hrs)")
        ax3.set_ylabel(r"$\sum S$ (mm)")
        ax4.set_xlabel(r"Thermal days ($^\circ$C days)")
        ax4.set_ylabel(self.metric + " (" + self.metric_units + "month$^{-1}$)")
        ax5.set_xlabel(r"$\sum P$ (mm)")
        ax5.set_ylabel(self.metric + " (" + self.metric_units + "month$^{-1}$)")
        ax6.set_xlabel(r"$\sum S$ (hrs)")
        ax6.set_ylabel(self.metric + " (" + self.metric_units + "month$^{-1}$)")

        # Save the figure
        fig.savefig("../Response_functions/response_" + self.cult + "_daily_3d.png", dpi=300, bbox_inches="tight")

    def post_prior_comp(self):

        labels = [r"norm", r"$\mu_t$", r"$\sigma_t$", r"$\mu_p$", r"$\sigma_p$", r"$\mu_s$", r"$\sigma_s$",
                  r"$\rho_tp$", r"$\rho_ts$", r"$\rho_ps$", r"$log_f$"]
        fig = corner.corner(self.flat_samples, show_titles=True, labels=labels, plot_datapoints=True,
                            quantiles=[0.16, 0.5, 0.84])

        fig.savefig("../model_performance/Corners/corner_" + self.cult + "_daily_3d.png", bbox_inches="tight")

        plt.close(fig)
