import datetime
import os
from os.path import abspath, dirname, join
import time
import warnings
# from multiprocessing import Pool

import emcee
import h5py
import numpy as np
import seaborn as sns
from utilities import extract_cultivar
from weather_extraction import read_or_create


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

    def __init__(self, cultivar, region_tol=0.25,
                 weather=("temperature", "rainfall", "sunshine"),
                 metric="yield",
                 metric_units="t/Ha", extract_flag=False, initial_guess=(
                        1200, 100, 700, 150, 1500, 150, -0.1, 0.1, -0.1),
                 initial_spread=(200, 150, 200, 150, 250, 150, 2, 2, 2),
                 seed_generator=create_seed):

        start = time.time()
        self.cult = cultivar
        self.tol = region_tol
        self.extract_flag = extract_flag
        self.initial_guess = initial_guess
        self.initial_spread = initial_spread
        self.weather = weather
        self.metric = metric
        self.metric_units = metric_units
        self.seed_generator = seed_generator

        # Extract imput data from cultivar csv's (see utilities)
        (
            self.reg_lats, self.reg_longs, self.reg_yrs,
            self.ripe_days, self.yield_data, self.sow_days, self.sow_months
        ) = extract_cultivar(cultivar)

        self.sow_year = self.reg_yrs - 1
        self.reg_keys = self.get_day_keys()
        self.reg_mth_keys = self.get_month_keys()
        self.reg_pred = np.zeros_like(self.yield_data)

        self.wthr_dict = {}
        self.wthr_anom_dict = {}
        self.mean_params = {}
        self.samples = {}

        self.fit = None
        self.resi = None
        self.norm = 16

        # Internal variables for speed
        self.norm_coeff = np.log(1 / ((2 * np.pi) ** (1 / 2)))
        self.log_initial_spread = [np.log(i) for i in initial_spread]

        weather_data = read_or_create(
            cultivar, self.reg_lats, self.reg_longs, self.sow_year,
            self.reg_keys, self.reg_mth_keys, self.tol)
        (
            self.temp_min, self.temp_max, self.precip, self.precip_anom,
            self.sun, self.sun_anom
        ) = weather_data

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
        p0 = self.seed_generator(nwalkers, ndim, self.initial_guess)

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
        print(f'{[f"{r:.3f}" for r in resi]} percent residual')
        print(f'Residual Mean: {np.mean(resi):.3f}')
        print(f'Residual Median: {np.median(resi):.3f}')

        self.resi = resi
