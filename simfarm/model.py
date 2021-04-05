import os
from os.path import abspath, dirname
import time
import warnings
# from multiprocessing import Pool

import emcee
from functools import partial
import numpy as np
import seaborn as sns

from .stats import gdd_calc, gauss3d, log_probability_3d


PARENT_DIR = dirname(dirname(abspath(__file__)))

warnings.filterwarnings("ignore")
# Does this do anything? AA
os.environ["OMP_NUM_THREADS"] = "1"

sns.set_style("whitegrid")
sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12,
                             "axes.labelsize": 12})


def create_seed(num_walkers, dimensions, initial_guess):
    return np.random.randn(num_walkers, dimensions) * 0.001 + initial_guess


class cultivarModel:

    def __init__(self, cultivar, cultivar_data, cultivar_weather_data,
                 metric="yield",
                 metric_units="t/Ha", initial_guess=(
                        1200, 100, 700, 150, 1500, 150, -0.1, 0.1, -0.1),
                 initial_spread=(200, 150, 200, 150, 250, 150, 2, 2, 2),
                 seed_generator=create_seed):

        start = time.time()
        self.cult = cultivar
        self.reg_lats = cultivar_data["Lat"]
        self.reg_longs = cultivar_data["Long"]
        self.reg_yrs = cultivar_data["Year"]
        self.ripe_days = cultivar_data["Ripe Time"]
        self.yield_data = cultivar_data["Yield"]
        self.sow_days = cultivar_data["Sow Day"]
        self.sow_months = cultivar_data["Sow Month"]

        (
            self.temp_min, self.temp_max, self.precip, self.precip_anom,
            self.sun, self.sun_anom
        ) = cultivar_weather_data

        self.initial_guess = initial_guess
        self.initial_spread = initial_spread
        self.metric = metric
        self.metric_units = metric_units
        self.seed_generator = seed_generator

        self.reg_pred = np.zeros_like(self.yield_data)

        # self.wthr_dict = {}  # unused???
        # self.wthr_anom_dict = {}  # unused ??
        self.mean_params = {}
        self.samples = {}

        self.fit = None
        self.resi = None
        self.norm = 16

        # Internal variables for speed
        self.norm_coeff = np.log(1 / ((2 * np.pi) ** (1 / 2)))
        self.log_initial_spread = [np.log(i) for i in initial_spread]

        # Compute thermal days and total rainfall
        self.therm_days = gdd_calc(self.temp_min, self.temp_max)
        self.tot_precip = np.nansum(self.precip, axis=1)
        self.tot_sun = np.nansum(self.sun, axis=1)

        print(f"Input extracted in {(time.time() - start):.2} seconds")

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
        p0 = self.seed_generator(nwalkers, ndim, self.initial_guess)

        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                        partial(log_probability_3d, self),
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
        self.af = np.mean(af)
        print(f"Mean acceptance fraction: {self.af:.3f}")
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
        print("mu_t (mean temperature +/- error) = %.3f +/- %.3f" % (self.mean_params["mu_t"], mu_t_err))
        print("sig_t (standard deviation temperature +/- error) = %.3f +/- %.3f" % (self.mean_params["sig_t"], sig_t_err))
        print("mu_p (mean precipitation +/- error) = %.3f +/- %.3f" % (self.mean_params["mu_p"], mu_p_err))
        print("sig_p (standard deviation precipitation +/- error) = %.3f +/- %.3f" % (self.mean_params["sig_p"], sig_p_err))
        print("mu_s (mean sunshine +/- error) = %.3f +/- %.3f" % (self.mean_params["mu_s"], mu_s_err))
        print("sig_s (standard deviation sunshine +/- error) = %.3f +/- %.3f" % (self.mean_params["sig_s"], sig_s_err))
        print("rho_tp (temperature and precipitation correlation +/- error) = %.3f +/- %.3f" % (
            self.mean_params["rho_tp"], rho_tp_err))
        print("rho_ts (temperature and sunshine correlation +/- error) = %.3f +/- %.3f" % (
            self.mean_params["rho_ts"], rho_ts_err))
        print("rho_ps (precipitation and sunshine correlation +/- error) = %.3f +/- %.3f" % (
            self.mean_params["rho_ps"], rho_ps_err))

        # Calculate the predicted results
        preds = gauss3d(self.norm, self.predict_temp,
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
        print(f'{[f"{r:.3f}" for r in resi]} percent residual')
        print(f'Residual Mean: {np.mean(resi):.3f}')
        print(f'Residual Median: {np.median(resi):.3f}')

        self.resi = resi
