import datetime
import os
import time
import warnings
from multiprocessing import Pool

import emcee
import h5py
import numpy as np
import seaborn as sns
import utilities


warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

sns.set_style("whitegrid")
sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12,
                             "axes.labelsize": 12})


class cultivarModel:

    def __init__(self, cultivar, region_tol=0.25,
                 weather=("temperature", "rainfall", "sunshine"),
                 metric="yield",
                 metric_units="t/Ha", extract_flag=False, initial_guess=(
                        1200, 100, 700, 150, 1500, 150, -0.1, 0.1, -0.1),
                 initial_spread=(200, 150, 200, 150, 250, 150, 2, 2, 2)):

        start = time.time()

        # Get the inputs
        if cultivar != "All":
            data = utilities.extract_data("../example_data/"
                                          + cultivar + "_Data.csv")
            region_lats, region_longs, years, \
                ripe_days, yields, sow_day, sow_month = data
        else:
            yield_path = "../All_Cultivars_Spreadsheets/Yield.csv"
            ripetime_path = "../All_Cultivars_Spreadsheets/Ripe Time.csv"
            data = utilities.extract_data_allwheat(yield_path, ripetime_path)
            region_lats, region_longs, years, \
                ripe_days, yields, sow_day, sow_month = data

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

        self.norm = 16

        # Internal variables for speed
        self.norm_coeff = np.log(1 / ((2 * np.pi) ** (1 / 2)))
        self.log_initial_spread = [np.log(i) for i in initial_spread]

        # Open file
        try:
            filename = "../Climate_Data/Region_Climate_" + self.cult + ".hdf5"

            hdf = h5py.File(filename, "r")

            self.temp_max, self.temp_min = hdf["Temperature_Maximum"][...], \
                hdf["Temperature_Minimum"][...]
            print("Temperature Extracted")
            self.precip_anom, self.precip = hdf["Rainfall_Anomaly"][...], \
                hdf["Rainfall"][...]
            print("Rainfall Extracted")
            self.sun_anom, self.sun = hdf["Sunshine_Anomaly"][...], \
                hdf["Sunshine"][...]
            print("Sunshine Extracted")

            hdf.close()

        except KeyError:
            extract_flag = True
            print("Key not found")
        except OSError:
            extract_flag = True
            print(f"File {filename} not found")

        if extract_flag:
            print("Extracting meterological files")
            self.temp_max = self.get_temp("tempmax")
            self.temp_min = self.get_temp("tempmin")

            # Apply conditions from
            # https://ndawn.ndsu.nodak.edu/help-wheat-growing-degree-days.html
            self.temp_max[self.temp_max < 0] = 0
            self.temp_min[self.temp_min < 0] = 0

            print("Temperature Extracted")
            self.precip_anom, self.precip = self.get_weather_anomaly(
                weather[1])
            print("Rainfall Extracted")
            self.sun_anom, self.sun = self.get_weather_anomaly_monthly(
                weather[2])
            print("Sunshine Extracted")

            hdf = h5py.File(filename, "w")

            hdf.create_dataset("Temperature_Maximum",
                               shape=self.temp_max.shape,
                               dtype=self.temp_max.dtype,
                               data=self.temp_max, compression="gzip")
            hdf.create_dataset("Temperature_Minimum",
                               shape=self.temp_min.shape,
                               dtype=self.temp_min.dtype,
                               data=self.temp_min, compression="gzip")
            hdf.create_dataset("Rainfall", shape=self.precip.shape,
                               dtype=self.precip.dtype,
                               data=self.precip, compression="gzip")
            hdf.create_dataset("Rainfall_Anomaly",
                               shape=self.precip_anom.shape,
                               dtype=self.precip_anom.dtype,
                               data=self.precip_anom, compression="gzip")
            hdf.create_dataset("Sunshine", shape=self.sun.shape,
                               dtype=self.sun.dtype,
                               data=self.sun, compression="gzip")
            hdf.create_dataset("Sunshine_Anomaly", shape=self.sun_anom.shape,
                               dtype=self.sun_anom.dtype,
                               data=self.sun_anom, compression="gzip")

            hdf.close()

        # Compute thermal days and total rainfall
        self.therm_days = self.gdd_calc(self.temp_min, self.temp_max)
        self.tot_precip = np.nansum(self.precip, axis=1)
        self.tot_sun = np.nansum(self.sun, axis=1)

        print(f"Input extracted in {(time.time() - start):.2} seconds")

    @staticmethod
    def gdd_calc(tempmin, tempmax):

        gdd = np.zeros(tempmin.shape[0])
        for ind in range(tempmin.shape[1]):

            tmaxs = tempmax[:, ind]
            tmins = tempmin[:, ind]

            if np.sum(tmaxs) == np.sum(tmins) == 0:
                break

            tmaxs[np.logical_and(gdd < 395, tmaxs > 21)] = 21
            tmaxs[np.logical_and(gdd >= 395, tmaxs > 35)] = 35

            gdd += (tmaxs - tmins) / 2

        return gdd

    @staticmethod
    def extract_region(lat, long, region_lat, region_long, weather, tol):

        # Get the boolean array for points within tolerence
        bool_cond = np.logical_and(np.abs(lat - region_lat) < tol,
                                   np.abs(long - region_long) < tol)

        # Get the extracted region
        ex_reg = weather[bool_cond]

        # Remove any nan and set them to 0 these correspond to ocean
        ex_reg = ex_reg[ex_reg < 1e8]

        if ex_reg.size == 0:
            print(f"Region not in coords: {region_lat} {region_long}")
            return np.nan
        else:
            return np.mean(ex_reg)

    @staticmethod
    def gauss2d_resp(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho):

        t_term = ((t - mu_t) / sig_t) ** 2
        p_term = ((p - mu_p) / sig_p) ** 2
        tp_term = 2 * rho * (t - mu_t) * (p - mu_p) / (sig_t * sig_p)

        dy = norm * np.exp(-(1 / (2 - 2 * rho * rho))
                           * (t_term + p_term - tp_term))

        return dy

    def get_day_keys(self):

        # Initialise the dictionary to hold keys
        sow_dict = {}

        # Loop over regions
        for regind, (lat, long, sow_yr) in enumerate(
                zip(self.reg_lats, self.reg_longs, self.sow_year)):

            # Initialise this regions entry
            sow_dict.setdefault(str(lat) + "." + str(long), {})

            # Extract this years sowing date and ripening time in days
            sow_day = self.sow_days[regind]
            sow_month = self.sow_months[regind]
            ripe_time = self.ripe_days[regind]
            sow_date = datetime.date(year=sow_yr, month=int(sow_month),
                                     day=int(sow_day))

            # Initialise this region"s dictionary entry
            hdf_keys = np.empty(ripe_time + 1, dtype=object)

            # Loop over months between sowing and ripening
            for nday in range(ripe_time + 1):
                # Compute the correct month number for this month
                key_date = sow_date + datetime.timedelta(days=nday)

                # Append this key to the dictionary under this
                # region in chronological order
                hdf_keys[nday] = str(
                    key_date.year) + "_%03d" % key_date.month + "_%04d" % key_date.day

            # Assign keys to dictionary
            sow_dict[str(lat) + "." + str(long)][str(sow_yr)] = hdf_keys

        return sow_dict

    def get_month_keys(self):

        # Initialise the dictionary to hold keys
        sow_dict = {}

        # Loop over regions
        for regind, (lat, long, sow_yr) in enumerate(
                zip(self.reg_lats, self.reg_longs, self.sow_year)):

            # Initialise this regions entry
            sow_dict.setdefault(str(lat) + "." + str(long), {})

            # Extract this years sowing date and ripening time in days
            sow_day = self.sow_days[regind]
            sow_month = self.sow_months[regind]
            ripe_time = self.ripe_days[regind]
            sow_date = datetime.date(year=sow_yr, month=int(sow_month),
                                     day=int(sow_day))

            # Initialise this region"s dictionary entry
            hdf_keys = []

            # Loop over months between sowing and ripening
            for ndays in range(ripe_time + 1):
                # Compute the correct month number for this month
                key_date = sow_date + datetime.timedelta(days=ndays)

                # Append this key to the dictionary under this
                # region in chronological order
                hdf_keys.append(str(key_date.year) + "_%03d" % key_date.month)

            # Assign keys to dictionary
            sow_dict[str(lat) + "." + str(long)][str(sow_yr)] = np.unique(
                hdf_keys)

        return sow_dict

    def get_weather_anomaly(self, weather):

        hdf = h5py.File("../SimFarm2030_" + weather + ".hdf5", "r")

        # Get the mean weather data for each month of the year
        uk_monthly_mean = hdf["all_years_mean"][...]

        lats = hdf["Latitude_grid"][...]
        longs = hdf["Longitude_grid"][...]

        done_wthr = {}

        # Loop over regions
        anom = np.full((len(self.reg_lats), 400), np.nan)
        wthr = np.full((len(self.reg_lats), 400), np.nan)
        for llind, (lat, long, year) in enumerate(
                zip(self.reg_lats, self.reg_longs, self.sow_year)):

            hdf_keys = self.reg_keys[str(lat) + "." + str(long)][str(year)]
            year_loc = f"{lat}_{long}_{year}"

            if year_loc in done_wthr:
                if tuple(hdf_keys) in done_wthr[year_loc]:
                    print(f"Already extracted {year_loc}")
                    wthr[llind, :] = done_wthr[year_loc][tuple(hdf_keys)]
                    continue

            # Initialise arrays to hold results
            key_ind = 0
            for key in hdf_keys:
                year, month, day = key.split("_")

                wthr_grid = hdf[key]["daily_grid"][...]

                ex_reg = self.extract_region(lats, longs, lat, long, wthr_grid,
                                             self.tol)

                # If year is within list of years extract the relevant data
                wthr[llind, key_ind] = ex_reg
                anom[llind, key_ind] = ex_reg - uk_monthly_mean[int(day) - 1]
                key_ind += 1

            done_wthr.setdefault(year_loc, {})[
                tuple(hdf_keys)] = wthr[llind, :]

        # Assign weather data to variable
        self.wthr_anom_dict[weather] = anom

        hdf.close()

        return anom, wthr

    def get_temp(self, weather):
        print(f'Getting the locations for new cultivar: {self.cult}')
        hdf = h5py.File("../SimFarm2030_" + weather + ".hdf5", "r")

        lats = hdf["Latitude_grid"][...]
        longs = hdf["Longitude_grid"][...]

        done_wthr = {}

        # Loop over regions
        print(f'Getting the temperature for those locations: {self.cult}')
        wthr = np.zeros((len(self.reg_lats), 400))
        for llind, (lat, long, year) in enumerate(
                zip(self.reg_lats, self.reg_longs, self.sow_year)):

            hdf_keys = self.reg_keys[str(lat) + "." + str(long)][str(year)]
            year_loc = f"{lat}_{long}_{year}"

            if year_loc in done_wthr:
                if tuple(hdf_keys) in done_wthr[year_loc]:
                    print("Already extracted {year_loc}")
                    wthr[llind, :] = \
                        done_wthr[year_loc][tuple(hdf_keys)]
                    continue

            # Initialise arrays to hold results
            print(f'Initialising array: {llind}')
            key_ind = 0
            for key in hdf_keys:
                year, month, day = key.split("_")

                wthr_grid = hdf[key]["daily_grid"][...]

                ex_reg = self.extract_region(lats, longs, lat, long, wthr_grid,
                                             self.tol)

                # If year is within list of years extract the relevant data
                wthr[llind, key_ind] = ex_reg
                key_ind += 1

            done_wthr.setdefault(year_loc, {})[tuple(hdf_keys)] = wthr[llind, :]

        hdf.close()

        return wthr

    def get_weather_anomaly_monthly(self, weather):

        hdf = h5py.File("../SimFarm2030_" + weather + ".hdf5", "r")

        # Get the mean weather data for each month of the year
        uk_monthly_mean = hdf["all_years_mean"][...]

        lats = hdf["Latitude_grid"][...]
        longs = hdf["Longitude_grid"][...]

        # Loop over regions
        anom = np.full((len(self.reg_lats), 15), np.nan)
        wthr = np.full((len(self.reg_lats), 15), np.nan)
        for llind, (lat, long, year) in enumerate(
                zip(self.reg_lats, self.reg_longs, self.sow_year)):

            hdf_keys = self.reg_mth_keys[str(lat) + "." + str(long)][str(year)]

            # Initialise arrays to hold results
            key_ind = 0
            for key in hdf_keys:
                year, month = key.split("_")

                wthr_grid = hdf[key]["monthly_grid"][...]

                ex_reg = self.extract_region(lats, longs, lat, long, wthr_grid,
                                             self.tol)

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

        dy = norm * np.exp(
            -0.5 / (1 - rho * rho) * (t_term + p_term - tp_term))

        return dy

    @staticmethod
    def gauss3d(norm, t, mu_t, sig_t, p, mu_p, sig_p, s, mu_s, sig_s, rho_tp,
                rho_ts, rho_ps):

        dy = norm * np.exp(-(0.5 * 1 / (
                    1 - np.square(rho_tp) - np.square(rho_ts) - np.square(
                        rho_ps)
                    + 2 * rho_tp * rho_ts * rho_ps))
                           * (np.square((t - mu_t) / sig_t)
                              + np.square((p - mu_p) / sig_p) + np.square(
                                (s - mu_s) / sig_s)
                              + 2 * ((t - mu_t) * (p - mu_p) * (
                                rho_ts * rho_ps - rho_tp) / (sig_t * sig_p)
                                     + (t - mu_t) * (s - mu_s) * (
                                                 rho_tp * rho_ts - rho_ps)
                                     / (sig_t * sig_s) + (p - mu_p) * (
                                                 s - mu_s)
                                     * (rho_tp * rho_ts - rho_ps) / (
                                                 sig_s * sig_p))))

        return dy

    @staticmethod
    def normpdf(x, loc, scale, logscale, coeff):

        u = (x - loc) / scale

        y = coeff - logscale - (u * u / 2)

        return y

    def log_likelihood_3d(self, theta, t, p, s, y, yerr):

        # Extract parameter values
        mu_t, sig_t, mu_p, sig_p, mu_s, sig_s, rho_tp, rho_ts, rho_ps = theta

        # Define model
        model = self.gauss3d(self.norm, t, mu_t, sig_t, p, mu_p, sig_p,
                             s, mu_s, sig_s, rho_tp, rho_ts, rho_ps)

        sigma2 = yerr ** 2

        return -0.5 * np.sum((y - model) ** 2 / sigma2)

    def log_prior_3d(self, theta):

        # Extract parameter values
        mu_t, sig_t, mu_p, sig_p, mu_s, sig_s, rho_tp, rho_ts, rho_ps = theta

        # The only parameters with a lower bound are norm and the sigmas
        cond = (500 < mu_t < 2000 and 0 < sig_t < 5000
                and 300 < mu_p < 2000 and 0 < sig_p < 5000
                and 500 < mu_s < 2500 and 0 < sig_s < 5000
                and -1 <= rho_tp <= 1 and -1 <= rho_ts <= 1
                and -1 <= rho_ps <= 1)
        if cond:

            # Define ln(prior) for each prior
            mut_lnprob = self.normpdf(mu_t,
                                      loc=self.initial_guess[0],
                                      scale=self.initial_spread[0],
                                      logscale=self.log_initial_spread[0],
                                      coeff=self.norm_coeff)
            sigt_lnprob = self.normpdf(sig_t,
                                       loc=self.initial_guess[1],
                                       scale=self.initial_spread[1],
                                       logscale=self.log_initial_spread[1],
                                       coeff=self.norm_coeff)
            mup_lnprob = self.normpdf(mu_p,
                                      loc=self.initial_guess[2],
                                      scale=self.initial_spread[2],
                                      logscale=self.log_initial_spread[2],
                                      coeff=self.norm_coeff)
            sigp_lnprob = self.normpdf(sig_p,
                                       loc=self.initial_guess[3],
                                       scale=self.initial_spread[3],
                                       logscale=self.log_initial_spread[3],
                                       coeff=self.norm_coeff)
            mus_lnprob = self.normpdf(mu_s,
                                      loc=self.initial_guess[4],
                                      scale=self.initial_spread[4],
                                      logscale=self.log_initial_spread[4],
                                      coeff=self.norm_coeff)
            sigs_lnprob = self.normpdf(sig_s,
                                       loc=self.initial_guess[5],
                                       scale=self.initial_spread[5],
                                       logscale=self.log_initial_spread[5],
                                       coeff=self.norm_coeff)
            rhotp_lnprob = self.normpdf(rho_tp,
                                        loc=self.initial_guess[6],
                                        scale=self.initial_spread[6],
                                        logscale=self.log_initial_spread[6],
                                        coeff=self.norm_coeff)
            rhots_lnprob = self.normpdf(rho_ts,
                                        loc=self.initial_guess[7],
                                        scale=self.initial_spread[7],
                                        logscale=self.log_initial_spread[7],
                                        coeff=self.norm_coeff)
            rhops_lnprob = self.normpdf(rho_ps,
                                        loc=self.initial_guess[8],
                                        scale=self.initial_spread[8],
                                        logscale=self.log_initial_spread[8],
                                        coeff=self.norm_coeff)

            return mut_lnprob + sigt_lnprob + mup_lnprob \
                + sigp_lnprob + mus_lnprob + sigs_lnprob \
                + rhotp_lnprob + rhots_lnprob + rhops_lnprob
            # return 0
        else:
            return -np.inf

    def log_probability_3d(self, theta, t, p, s, y, yerr):
        lp = self.log_prior_3d(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_3d(theta, t, p, s, y, yerr)

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

        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                        self.log_probability_3d,
                                        args=(self.train_temp, self.train_rain,
                                              self.train_sun,
                                              self.train_yields, self.yerr))

        # Run 1000 steps(random choice) as a burn-in.
        print(
            "Burning in 1000 steps... "
            "letting the walkers explore "
            "the parameter space before starting")
        pos, prob, state = sampler.run_mcmc(p0, 1000)

        # Reset the chain to remove the burn-in samples.
        # https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
        sampler.reset()

        print("Running MCMC ...")
        pos, prob, state = sampler.run_mcmc(p0, nsample, progress=True,
                                            rstate0=state)

        self.model = sampler

        # Print out the mean acceptance fraction. In general,
        # acceptance_fraction has an entry for each walker so,
        # in this case, it is a 250-dimensional vector.
        af = sampler.acceptance_fraction
        print(f"Mean acceptance fraction: {np.mean(af):.3f}")
        af_msg = """As a rule of thumb, the acceptance fraction (af) should be
                                    between 0.2 and 0.5
                    If af < 0.2 decrease the a parameter
                    If af > 0.5 increase the a parameter
                    """

        print(af_msg)

        flat_samples = sampler.get_chain(discard=1000, thin=100, flat=True)
        self.flat_samples = flat_samples

        # Extract fitted parameters
        d = self.mean_params
        maxprob_indice = np.argmax(prob)
        self.maxprob_params = pos[maxprob_indice]
        self.fitted_params = np.median(flat_samples, axis=0)
        d["mu_t"], d["sig_t"], d["mu_p"], d["sig_p"], d["mu_s"], d["sig_s"], d[
            "rho_tp"], d["rho_ts"], d["rho_ps"] = self.fitted_params

        # Extract the samples
        d = self.samples
        d["mu_t"], d["sig_t"], d["mu_p"], d["sig_p"], d["mu_s"], d["sig_s"], d[
            "rho_tp"], d["rho_ts"], d["rho_ps"] = [flat_samples[:, i] for i in
                                                   range(ndim)]

        # Extract the errors on the fitted parameters
        self.param_errors = np.std(flat_samples, axis=0)
        mu_t_err, sig_t_err, mu_p_err, sig_p_err, mu_s_err, sig_s_err, rho_tp_err, rho_ts_err, rho_ps_err = self.param_errors

        print("================ Model Parameters ================")
        print("mu_t = %.3f +/- %.3f" % (self.mean_params["mu_t"], mu_t_err))
        print("sig_t = %.3f +/- %.3f" % (self.mean_params["sig_t"], sig_t_err))
        print("mu_p = %.3f +/- %.3f" % (self.mean_params["mu_p"], mu_p_err))
        print("sig_p = %.3f +/- %.3f" % (self.mean_params["sig_p"], sig_p_err))
        print("mu_s = %.3f +/- %.3f" % (self.mean_params["mu_s"], mu_s_err))
        print("sig_s = %.3f +/- %.3f" % (self.mean_params["sig_s"], sig_s_err))
        print("rho_tp = %.3f +/- %.3f" % (
            self.mean_params["rho_tp"], rho_tp_err))
        print("rho_ts = %.3f +/- %.3f" % (
            self.mean_params["rho_ts"], rho_ts_err))
        print("rho_ps = %.3f +/- %.3f" % (
            self.mean_params["rho_ps"], rho_ps_err))

        # Calculate the predicted results
        preds = self.gauss3d(self.norm, self.predict_temp,
                             self.mean_params["mu_t"],
                             self.mean_params["sig_t"], self.predict_rain,
                             self.mean_params["mu_p"],
                             self.mean_params["sig_p"], self.predict_sun,
                             self.mean_params["mu_s"],
                             self.mean_params["sig_s"],
                             self.mean_params["rho_tp"],
                             self.mean_params["rho_ts"],
                             self.mean_params["rho_ps"])

        self.preds = preds

        # Calculate the percentage residual
        resi = (1 - (preds / self.predict_yields)) * 100
        print(resi)
        print(np.mean(resi))
        print(np.median(resi))

        self.resi = resi
