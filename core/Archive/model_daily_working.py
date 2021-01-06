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

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

sns.set_style("whitegrid")
sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12})


class cultivarModel:

    def __init__(self, cultivar, region_tol=0.25, weather=("temperature", "rainfall"), metric="yield",
                 metric_units="t/Ha", extract_flag=False, initial_guess=(10, 3375, 380, 705, 220, 0),
                 initial_spread=(15, 500, 300, 400, 300, 1)):

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

            hdf = h5py.File("../Climate_Data/Region_Climate_" + self.cult + "_daily.hdf5", "r")

            self.temp_anom, self.temp = hdf["Temperature_Anomaly"][...], hdf["Temperature"][...]
            self.temp[self.temp <= 0] = np.nan
            print("Temperature Extracted")
            self.precip_anom, self.precip = hdf["Rainfall_Anomaly"][...], hdf["Rainfall"][...]
            print("Rainfall Extracted")

            hdf.close()

        except KeyError:
            self.extract_flag = True
        except OSError:
            self.extract_flag = True

        if self.extract_flag:

            self.temp_anom, self.temp = self.get_weather_anomaly(weather[0])
            self.temp[self.temp <= 0] = np.nan
            print("Temperature Extracted")
            self.precip_anom, self.precip = self.get_weather_anomaly(weather[1])
            print("Rainfall Extracted")

            hdf = h5py.File("../Climate_Data/Region_Climate_" + self.cult + "_daily.hdf5", "w")

            hdf.create_dataset("Temperature", shape=self.temp.shape, dtype=self.temp.dtype,
                               data=self.temp, compression="gzip")
            hdf.create_dataset("Temperature_Anomaly", shape=self.temp_anom.shape, dtype=self.temp_anom.dtype,
                               data=self.temp_anom, compression="gzip")
            hdf.create_dataset("Rainfall", shape=self.precip.shape, dtype=self.precip.dtype,
                               data=self.precip, compression="gzip")
            hdf.create_dataset("Rainfall_Anomaly", shape=self.precip_anom.shape, dtype=self.precip_anom.dtype,
                               data=self.precip_anom, compression="gzip")

            hdf.close()

        # Compute thermal days and total rainfall
        self.therm_days = np.nansum(self.temp, axis=1)
        self.tot_precip = np.nansum(self.precip, axis=1)

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
    def gauss3d(norm, t, mu_t, sig_t, p, mu_p, sig_p, s, mu_s, sig_s, rho_tp, rho_ts, rho_ps):

        dy = 0
        for mon in range(0, 12):
            dy += norm * np.exp(-(0.5 * 1 / (1 - np.square(rho_tp) - np.square(rho_ts) - np.square(rho_ps)
                                             + 2 * rho_tp * rho_ts * rho_ps))
                                * (np.square((t[mon] - mu_t) / sig_t)
                                   + np.square((p[mon] - mu_p) / sig_p) + np.square((s[mon] - mu_s) / sig_s)
                                   + 2 * ((t[mon] - mu_t) * (p[mon] - mu_p) * (rho_ts * rho_ps - rho_tp) / (
                                sig_t * sig_p)
                                          + (t[mon] - mu_t) * (s[mon] - mu_s) * (rho_tp * rho_ts - rho_ps)
                                          / (sig_t * sig_s) + (p[mon] - mu_p) * (s[mon] - mu_s)
                                          * (rho_tp * rho_ts - rho_ps) / (sig_s * sig_p))))

        return dy

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

    # @staticmethod
    # def gauss1d(x, norm, mu, sig):
    #
    #     exp_term = ((x - mu) / sig) ** 2
    #
    #     dy = norm * np.nansum(np.exp(-0.5 * exp_term), axis=1)
    #
    #     return dy

    def log_likelihood_2d(self, theta, t, p, y, yerr):

        # Extract initial guesses
        norm, mu_t, sig_t, mu_p, sig_p, rho = theta

        # Define model
        model = self.gauss2d(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho)

        sigma2 = yerr ** 2

        return -0.5 * np.sum((y - model) ** 2 / sigma2)

    # def log_likelihood_1d(self, theta, t, y, yerr):
    #
    #     # Extract initial guesses
    #     norm, mu, sig = theta
    #
    #     # Define model
    #     model = self.gauss1d(t, norm, mu, sig)
    #
    #     sigma2 = yerr ** 2
    #
    #     return -0.5 * np.sum((y - model) ** 2 / sigma2)

    def log_prior_2d(self, theta):

        # Extract parameters from vector
        norm, mu_t, sig_t, mu_p, sig_p, rho = theta

        # The only parameters with a lower bound are norm and the sigmas
        cond = (0 < norm < 20
                and 2000 < mu_t < 4050 and 0 < sig_t < 10000
                and 350 < mu_p < 1700 and 0 < sig_p < 10000
                and -1 <= rho <= 1)
        if cond:

            # # Define ln(prior) for each prior
            norm_lnprob = np.log(stat.norm.pdf(norm, loc=self.initial_guess[0], scale=self.initial_spread[0]))
            mut_lnprob = np.log(stat.norm.pdf(mu_t, loc=self.initial_guess[1], scale=self.initial_spread[1]))
            sigt_lnprob = np.log(stat.norm.pdf(sig_t, loc=self.initial_guess[2], scale=self.initial_spread[2]))
            mup_lnprob = np.log(stat.norm.pdf(mu_p, loc=self.initial_guess[3], scale=self.initial_spread[3]))
            sigp_lnprob = np.log(stat.norm.pdf(sig_p, loc=self.initial_guess[4], scale=self.initial_spread[4]))
            rho_lnprob = np.log(stat.norm.pdf(rho, loc=self.initial_guess[5], scale=self.initial_spread[5]))

            return norm_lnprob + mut_lnprob + sigt_lnprob + mup_lnprob + sigp_lnprob + rho_lnprob
            # return 0
        else:
            return -np.inf

    # def log_prior_1d(self, theta, inds):
    #
    #     # Extract parameters from vector
    #     norm, mu, sig = theta
    #
    #     # The only parameters with a lower bound are norm and the sigmas
    #     cond = (0 < norm < 20 and 0 < mu < 50000 and 0 < sig < 10000)
    #     if cond:
    #
    #         # # Define ln(prior) for each prior
    #         norm_lnprob = np.log(
    #             stat.norm.pdf(norm, loc=self.initial_guess[inds[0]], scale=self.initial_spread[inds[0]]))
    #         mu_lnprob = np.log(stat.norm.pdf(mu, loc=self.initial_guess[inds[1]], scale=self.initial_spread[inds[1]]))
    #         sig_lnprob = np.log(stat.norm.pdf(sig, loc=self.initial_guess[inds[2]], scale=self.initial_spread[inds[2]]))
    #
    #         return norm_lnprob + mu_lnprob + sig_lnprob
    #         # return 0
    #     else:
    #         return -np.inf

    def log_probability_2d(self, theta, t, p, y, yerr):
        lp = self.log_prior_2d(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_2d(theta, t, p, y, yerr)

    # def log_probability_1d(self, theta, x, y, yerr, inds):
    #     lp = self.log_prior_1d(theta, inds)
    #     if not np.isfinite(lp):
    #         return -np.inf
    #     return lp + self.log_likelihood_1d(theta, x, y, yerr)

    def train_model(self, nsample=5000, nwalkers=500):

        temp = self.therm_days
        rain = self.tot_precip

        yields = self.yield_data
        yerr = np.std(self.yield_data)

        ndim = len(self.initial_guess)

        p0 = np.random.randn(nwalkers, ndim) * 0.0001 + self.initial_guess

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_2d,
                                            args=(temp, rain, yields, yerr), pool=pool, threads=8)

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

        flat_samples = sampler.get_chain(discard=500, thin=100, flat=True)
        self.flat_samples = flat_samples

        # Extract fitted parameters
        d = self.mean_params
        maxprob_indice = np.argmax(prob)
        self.maxprob_params = pos[maxprob_indice]
        self.fitted_params = np.median(flat_samples, axis=0)
        d["norm"], d['mu_t'], d["sig_t"], d['mu_p'], d["sig_p"], d["rho"] = self.fitted_params

        # Extract the samples
        d = self.samples
        d["norm"], d['mu_t'], d["sig_t"], d['mu_p'], d["sig_p"], d["rho"] = [flat_samples[:, i] for i in range(ndim)]

        # Extract the errors on the fitted parameters
        self.param_errors = np.std(flat_samples, axis=0)
        norm_err, mu_t_err, sig_t_err, mu_p_err, sig_p_err, rho_err = self.param_errors

        print("================ Model Parameters ================")
        print("norm = %.3f +/- %.3f" % (self.mean_params['norm'], norm_err))
        print("mu_t = %.3f +/- %.3f" % (self.mean_params['mu_t'], mu_t_err))
        print("sig_t = %.3f +/- %.3f" % (self.mean_params["sig_t"], sig_t_err))
        print("mu_p = %.3f +/- %.3f" % (self.mean_params['mu_p'], mu_p_err))
        print("sig_p = %.3f +/- %.3f" % (self.mean_params["sig_p"], sig_p_err))
        print("rho = %.3f +/- %.3f" % (self.mean_params["rho"], rho_err))

    def train_and_validate_model(self, split=0.7, nsample=5000, nwalkers=500):

        # Compute the ratio to split by
        size = self.therm_days.shape[0]
        predict_size = int(size * (1 - split))

        rand_inds = np.random.choice(np.arange(size), predict_size)
        okinds = np.zeros(size, dtype=bool)
        okinds[rand_inds] = True

        train_temp = self.therm_days[~okinds]
        train_rain = self.tot_precip[~okinds]
        train_yields = self.yield_data[~okinds]
        predict_temp = self.therm_days[okinds]
        predict_rain = self.tot_precip[okinds]
        predict_yields = self.yield_data[okinds]
        yerr = np.std(self.yield_data)

        ndim = len(self.initial_guess)

        p0 = np.random.randn(nwalkers, ndim) * 0.001 + self.initial_guess

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_2d,
                                        args=(train_temp, train_rain, train_yields, yerr))

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
        d["norm"], d['mu_t'], d["sig_t"], d['mu_p'], d["sig_p"], d["rho"] = self.fitted_params

        # Extract the samples
        d = self.samples
        d["norm"], d['mu_t'], d["sig_t"], d['mu_p'], d["sig_p"], d["rho"] = [flat_samples[:, i] for i in range(ndim)]

        # Extract the errors on the fitted parameters
        self.param_errors = np.std(flat_samples, axis=0)
        norm_err, mu_t_err, sig_t_err, mu_p_err, sig_p_err, rho_err = self.param_errors

        print("================ Model Parameters ================")
        print("norm = %.3f +/- %.3f" % (self.mean_params['norm'], norm_err))
        print("mu_t = %.3f +/- %.3f" % (self.mean_params['mu_t'], mu_t_err))
        print("sig_t = %.3f +/- %.3f" % (self.mean_params["sig_t"], sig_t_err))
        print("mu_p = %.3f +/- %.3f" % (self.mean_params['mu_p'], mu_p_err))
        print("sig_p = %.3f +/- %.3f" % (self.mean_params["sig_p"], sig_p_err))
        print("rho = %.3f +/- %.3f" % (self.mean_params["rho"], rho_err))

        # Calculate the predicted results
        preds = self.gauss2d(predict_temp, self.mean_params['norm'], self.mean_params['mu_t'],
                             self.mean_params['sig_t'], predict_rain, self.mean_params['mu_p'],
                             self.mean_params['sig_p'], self.mean_params["rho"])
        print(preds)

        # Calculate the percentage residual
        resi = (1 - preds / predict_yields) * 100
        print(resi)
        print(np.mean(resi))
        print(np.median(resi))
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

        fig.savefig("../model_performance/Validation/" + self.cult + ".png", bbox_inches="tight")

    def plot_walkers(self):

        ndim = len(self.initial_guess)

        for i in range(ndim):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            res = ax.plot(self.model.chain[:, :, i].T, "-", color="k", alpha=0.3)
            fig.savefig(f"../model_performance/Chains/samplerchain_{i}_" + self.cult + "_daily.png", bbox_inches="tight")
            plt.close(fig)

    def plot_response(self):

        # Create arrays to evaluate response function at
        eval_t = np.linspace(0, 6000, 1000)
        eval_p = np.linspace(0, 3500, 1000)

        # Get grid of values
        tt, pp = np.meshgrid(eval_t, eval_p)

        # Compute temperature response
        t_resp = self.gauss2d_resp(eval_t, self.mean_params["norm"], self.mean_params["mu_t"],
                                   self.mean_params["sig_t"], 0, self.mean_params["mu_p"],
                                   self.mean_params["sig_p"], self.mean_params["rho"])

        # Compute precipitation response
        p_resp = self.gauss2d_resp(0, self.mean_params["norm"], self.mean_params["mu_t"],
                                   self.mean_params["sig_t"], eval_p, self.mean_params["mu_p"],
                                   self.mean_params["sig_p"], self.mean_params["rho"])

        # Compute the response grid
        resp_grid = self.gauss2d_resp(tt, self.mean_params["norm"], self.mean_params["mu_t"],
                                      self.mean_params["sig_t"], pp, self.mean_params["mu_p"],
                                      self.mean_params["sig_p"], self.mean_params["rho"])

        # Set up figure
        fig = plt.figure(figsize=(9, 9.6))
        gs = gridspec.GridSpec(3, 2)
        gs.update(wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(gs[:2, :])
        ax2 = fig.add_subplot(gs[2, 0])
        ax3 = fig.add_subplot(gs[2, 1])

        # Plot the response functions
        cba = ax1.pcolormesh(eval_t, eval_p, resp_grid)

        cbar = fig.colorbar(cba, ax=ax1)
        cbar.ax.set_ylabel(self.metric + " (" + self.metric_units + "month$^{-1}$)")

        ax2.plot(eval_t, t_resp)
        ax3.plot(eval_p, p_resp)

        # Label axes
        ax1.set_xlabel(r"Thermal days ($^\circ$C days)")
        ax1.set_ylabel(r"$\sum P$ (mm)")
        ax2.set_xlabel(r"Thermal days ($^\circ$C days)")
        ax2.set_ylabel(self.metric + " (" + self.metric_units + "month$^{-1}$)")
        ax3.set_xlabel(r"$\sum P$ (mm)")
        ax3.set_ylabel(self.metric + " (" + self.metric_units + "month$^{-1}$)")

        # Save the figure
        fig.savefig("../Response_functions/response_" + self.cult + "_daily.png", dpi=300, bbox_inches="tight")

        return eval_t, eval_p, t_resp, p_resp, resp_grid

    def post_prior_comp(self):

        labels = [r"norm", r"$\mu_t$", r"$\sigma_t$", r"$\mu_p$", r"$\sigma_p$", r"$\rho$", "lagt", "lagp"]
        fig = corner.corner(self.flat_samples, show_titles=True, labels=labels, plot_datapoints=True,
                            quantiles=[0.16, 0.5, 0.84])

        fig.savefig("../model_performance/Corners/corner_" + self.cult + "_daily.png", bbox_inches="tight")

        plt.close(fig)
