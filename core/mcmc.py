import numpy as np
import emcee
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt


def gauss2d_daily(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho, lagt=1, lagp=4):

    t = pd.DataFrame(t).rolling(window=lagt, min_periods=1, axis=1).mean().values
    p = pd.DataFrame(p).rolling(window=lagp, min_periods=1, axis=1).mean().values

    t_term = ((t - mu_t) / sig_t) ** 2
    p_term = ((p - mu_p) / sig_p) ** 2
    tp_term = 2 * rho * (t - mu_t) * (p - mu_p) / (sig_t * sig_p)

    dy = norm * np.nansum(np.exp(-0.5 / (1 - rho * rho) * (t_term + p_term - tp_term)), axis=1)

    return dy


def gauss2d(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho):

    t_term = ((t - mu_t) / sig_t) ** 2
    p_term = ((p - mu_p) / sig_p) ** 2
    tp_term = 2 * rho * (t - mu_t) * (p - mu_p) / (sig_t * sig_p)

    dy = norm * np.nansum(np.exp(-0.5 / (1 - rho * rho) * (t_term + p_term - tp_term)), axis=1)

    return dy


def log_likelihood_daily(theta, t, p, y, yerr):
    
    # Extract initial guesses
    norm, mu_t, sig_t, mu_p, sig_p, rho = theta

    # Define model
    model = gauss2d_daily(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho, int(lagt), int(lagp))

    sigma2 = yerr ** 2

    return -0.5 * np.sum((y - model) ** 2 / sigma2)


def log_likelihood(theta, t, p, y, yerr):
    # Extract initial guesses
    norm, mu_t, sig_t, mu_p, sig_p, rho = theta

    # Define model
    model = gauss2d(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho)

    sigma2 = yerr ** 2

    return -0.5 * np.sum((y - model) ** 2 / sigma2)


def log_prior_daily(simfarm, theta):
    
    # Extract parameters from vector
    norm, mu_t, sig_t, mu_p, sig_p, rho = theta

    # The only parameters with a lower bound are norm and the sigmas
    cond = (0 < norm < 10 and -50 < mu_t < 50 and 0 < sig_t < 70
            and -200 < mu_p < 200 and 0 < sig_p < 700 and -1 <= rho <= 1)
    if cond:

        # # # Define ln(prior) for each prior
        # norm_lnprob = np.log(stat.halfnorm.pdf(norm, loc=simfarm.initial_guess[0], scale=simfarm.initial_spread[0]))
        # mut_lnprob = np.log(stat.norm.pdf(mu_t, loc=simfarm.initial_guess[1], scale=simfarm.initial_spread[1]))
        # sigt_lnprob = np.log(stat.halfnorm.pdf(sig_t, loc=simfarm.initial_guess[2], scale=simfarm.initial_spread[2]))
        # mup_lnprob = np.log(stat.norm.pdf(mu_p, loc=simfarm.initial_guess[3], scale=simfarm.initial_spread[3]))
        # sigp_lnprob = np.log(stat.halfnorm.pdf(sig_p, loc=simfarm.initial_guess[4], scale=simfarm.initial_spread[4]))
        # rho_lnprob = np.log(stat.norm.pdf(rho, loc=simfarm.initial_guess[5], scale=simfarm.initial_spread[5]))
        # lagt_lnprob = np.log(stat.halfnorm.pdf(lagt, loc=simfarm.initial_guess[6], scale=simfarm.initial_spread[6]))
        # lagp_lnprob = np.log(stat.halfnorm.pdf(lagp, loc=simfarm.initial_guess[7], scale=simfarm.initial_spread[7]))
        #
        # return (norm_lnprob + mut_lnprob + sigt_lnprob + mup_lnprob
        #         + sigp_lnprob + rho_lnprob + lagt_lnprob + lagp_lnprob)
        return 0
    else:
        return -np.inf


def log_prior(simfarm, theta):

    # Extract parameters from vector
    norm, mu_t, sig_t, mu_p, sig_p, rho = theta

    # The only parameters with a lower bound are norm and the sigmas
    cond = (0 < norm < 10 and -50 < mu_t < 50 and 0 < sig_t < 70
            and -200 < mu_p < 200 and 0 < sig_p < 700 and -1 <= rho <= 1)
    if cond:

        # # # Define ln(prior) for each prior
        # norm_lnprob = np.log(stat.halfnorm.pdf(norm, loc=simfarm.initial_guess[0], scale=simfarm.initial_spread[0]))
        # mut_lnprob = np.log(stat.norm.pdf(mu_t, loc=simfarm.initial_guess[1], scale=simfarm.initial_spread[1]))
        # sigt_lnprob = np.log(stat.halfnorm.pdf(sig_t, loc=simfarm.initial_guess[2], scale=simfarm.initial_spread[2]))
        # mup_lnprob = np.log(stat.norm.pdf(mu_p, loc=simfarm.initial_guess[3], scale=simfarm.initial_spread[3]))
        # sigp_lnprob = np.log(stat.halfnorm.pdf(sig_p, loc=simfarm.initial_guess[4], scale=simfarm.initial_spread[4]))
        # rho_lnprob = np.log(stat.norm.pdf(rho, loc=simfarm.initial_guess[5], scale=simfarm.initial_spread[5]))
        # lagt_lnprob = np.log(stat.halfnorm.pdf(lagt, loc=simfarm.initial_guess[6], scale=simfarm.initial_spread[6]))
        # lagp_lnprob = np.log(stat.halfnorm.pdf(lagp, loc=simfarm.initial_guess[7], scale=simfarm.initial_spread[7]))
        #
        # return (norm_lnprob + mut_lnprob + sigt_lnprob + mup_lnprob
        #         + sigp_lnprob + rho_lnprob + lagt_lnprob + lagp_lnprob)
        return 0
    else:
        return -np.inf


def log_probability_daily(simfarm, theta, t, p, y, yerr):

    lp = log_prior_daily(simfarm, theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_daily(theta, t, p, y, yerr)


def log_probability(simfarm, theta, t, p, y, yerr):

    lp = log_prior(simfarm, theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, p, y, yerr)


def train_model(simfarm, nsample=5000, nwalkers=500):

    temp = simfarm.temp_anom
    rain = simfarm.precip_anom

    yields = simfarm.yield_data
    yerr = np.std(simfarm.yield_data)

    ndim = len(simfarm.initial_guess)

    p0 = np.random.randn(nwalkers, ndim)

    with Pool() as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, simfarm.log_probability,
                                        args=(temp, rain, yields, yerr), pool=pool)

        # Run 200 steps as a burn-in.
        print("Burning in ...")
        pos, prob, state = sampler.run_mcmc(p0, 200)

        # Reset the chain to remove the burn-in samples.
        sampler.reset()

        print("Running MCMC ...")
        pos, prob, state = sampler.run_mcmc(p0, nsample, progress=True, rstate0=state)

    simfarm.model = sampler

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional
    # vector.
    af = sampler.acceptance_fraction
    print("Mean acceptance fraction:", np.mean(af))
    af_msg = """As a rule of thumb, the acceptance fraction (af) should be 
                                between 0.2 and 0.5
                If af < 0.2 decrease the a parameter
                If af > 0.5 increase the a parameter
                """

    print(af_msg)

    # Extract fitted parameters
    d = simfarm.mean_params
    maxprob_indice = np.argmax(prob)
    simfarm.fitted_params = pos[maxprob_indice]
    d["norm"], d["mu_t"], d["sig_t"], d["mu_p"], d["sig_p"], d["rho"] = simfarm.fitted_params

    # Extract the samples
    d = simfarm.samples
    d["norm"], d["mu_t"], d["sig_t"], d["mu_p"], d["sig_p"], d["rho"] = [sampler.flatchain[:, i] for i in range(ndim)]

    # Extract the errors on the fitted parameters
    simfarm.param_errors = np.std(sampler.flatchain, axis=0)
    norm_err, mu_t_err, sig_t_err, mu_p_err, sig_p_err, rho_err = simfarm.param_errors

    print("================ Model Parameters ================")
    print("norm = %.3f +/- %.3f" % (simfarm.mean_params["norm"], norm_err))
    print("mu_t = %.3f +/- %.3f" % (simfarm.mean_params["mu_t"], mu_t_err))
    print("sig_t = %.3f +/- %.3f" % (simfarm.mean_params["sig_t"], sig_t_err))
    print("mu_p = %.3f +/- %.3f" % (simfarm.mean_params["mu_p"], mu_p_err))
    print("sig_p = %.3f +/- %.3f" % (simfarm.mean_params["sig_p"], sig_p_err))
    print("rho = %.3f +/- %.3f" % (simfarm.mean_params["rho"], rho_err))

    tau = sampler.get_autocorr_time()
    print("Steps until initial start 'forgotten'", tau)


def train_and_validate_model(simfarm, split=0.7, nsample=5000, nwalkers=500):

    print("here")
    # Compute the ratio to split by
    size = simfarm.temp_anom.shape[0]
    predict_size = int(size * (1 - split))
    print("here")
    rand_inds = np.random.choice(np.arange(size), predict_size)
    okinds = np.zeros(size, dtype=bool)
    okinds[rand_inds] = True
    print("here")
    train_temp = simfarm.temp_anom[~okinds, :]
    train_rain = simfarm.precip_anom[~okinds, :]
    train_yields = simfarm.yield_data[~okinds]
    predict_temp = simfarm.temp_anom[okinds, :]
    predict_rain = simfarm.precip_anom[okinds, :]
    predict_yields = simfarm.yield_data[okinds]
    yerr = np.std(simfarm.yield_data)
    print("here")
    ndim = len(simfarm.initial_guess)
    print("here")
    p0 = np.random.randn(nwalkers, ndim) * 0.001 + simfarm.initial_guess
    print("here")
    with Pool() as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, simfarm.log_probability,
                                        args=(train_temp, train_rain, train_yields, yerr), pool=pool, threads=8)

        # Run 200 steps as a burn-in.
        print("Burning in ...")
        pos, prob, state = sampler.run_mcmc(p0, 500)

        # Reset the chain to remove the burn-in samples.
        sampler.reset()

        print("Running MCMC ...")
        pos, prob, state = sampler.run_mcmc(p0, nsample, progress=True, rstate0=state)

    simfarm.model = sampler

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional
    # vector.
    af = sampler.acceptance_fraction
    print("Mean acceptance fraction:", np.mean(af))
    af_msg = """As a rule of thumb, the acceptance fraction (af) should be 
                                between 0.2 and 0.5
                If af < 0.2 decrease the a parameter
                If af > 0.5 increase the a parameter
                """

    print(af_msg)

    # Extract fitted parameters
    d = simfarm.mean_params
    maxprob_indice = np.argmax(prob)
    simfarm.fitted_params = pos[maxprob_indice]
    d["norm"], d["mu_t"], d["sig_t"], d["mu_p"], d["sig_p"], d["rho"] = simfarm.fitted_params

    # Extract the samples
    d = simfarm.samples
    d["norm"], d["mu_t"], d["sig_t"], d["mu_p"], d["sig_p"], d["rho"] = [sampler.flatchain[:, i] for i in range(ndim)]

    # Extract the errors on the fitted parameters
    simfarm.param_errors = np.std(sampler.flatchain, axis=0)
    norm_err, mu_t_err, sig_t_err, mu_p_err, sig_p_err, rho_err = simfarm.param_errors

    print("================ Model Parameters ================")
    print("norm = %.3f +/- %.3f" % (simfarm.mean_params["norm"], norm_err))
    print("mu_t = %.3f +/- %.3f" % (simfarm.mean_params["mu_t"], mu_t_err))
    print("sig_t = %.3f +/- %.3f" % (simfarm.mean_params["sig_t"], sig_t_err))
    print("mu_p = %.3f +/- %.3f" % (simfarm.mean_params["mu_p"], mu_p_err))
    print("sig_p = %.3f +/- %.3f" % (simfarm.mean_params["sig_p"], sig_p_err))
    print("rho = %.3f +/- %.3f" % (simfarm.mean_params["rho"], rho_err))

    # Calculate the predicted results
    preds = gauss2d_daily(predict_temp, simfarm.mean_params["norm"], simfarm.mean_params["mu_t"],
                          simfarm.mean_params["sig_t"], predict_rain, simfarm.mean_params["mu_p"],
                          simfarm.mean_params["sig_p"], simfarm.mean_params["rho"])
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

    fig.savefig("../model_performance/validation_" + simfarm.cult + "_daily.png", bbox_inches="tight")

    # tau = sampler.get_autocorr_time()
    # print("Steps until initial start 'forgotten'", tau)