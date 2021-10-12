import numpy as np


def gdd_calc(tempmin, tempmax):

    gdd = np.zeros(tempmin.shape[0]) # shape of zeros
    for ind in range(tempmin.shape[1]): # why 2 axis, is one location

        tmaxs = tempmax[:, ind] # every index for each day/location
        tmins = tempmin[:, ind]

        if np.sum(tmaxs) == np.sum(tmins) == 0:
            break

        tmaxs[np.logical_and(gdd < 395, tmaxs > 21)] = 21  # will return a boolean, if gdd days under 395 and tmax above 21 then set to 21? why?
        tmaxs[np.logical_and(gdd >= 395, tmaxs > 35)] = 35 # why do you care what gdd is? when is 395 gdd

        gdd += (tmaxs - tmins) / 2 # can this work for value only or array

    return gdd


def gauss2d_resp(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho):

    t_term = ((t - mu_t) / sig_t) ** 2
    p_term = ((p - mu_p) / sig_p) ** 2
    tp_term = 2 * rho * (t - mu_t) * (p - mu_p) / (sig_t * sig_p)

    dy = norm * np.exp(
        -(1 / (2 - 2 * rho * rho))
        * (t_term + p_term - tp_term))

    return dy


def gauss2d(t, norm, mu_t, sig_t, p, mu_p, sig_p, rho):

    t_term = ((t - mu_t) / sig_t) ** 2
    p_term = ((p - mu_p) / sig_p) ** 2
    tp_term = 2 * rho * (t - mu_t) * (p - mu_p) / (sig_t * sig_p)

    dy = norm * np.exp(
        -0.5 / (1 - rho * rho) * (t_term + p_term - tp_term))

    return dy


def gauss3d(norm, t, mu_t, sig_t, p, mu_p, sig_p, s, mu_s, sig_s, rho_tp,
            rho_ts, rho_ps):
    '''t, p, s are the validation 30%(split/size) self.predict_temp/self.predict_rain/self.predict_sun
    of size (34,0) (skyfalls case) so although mean_params are scalar output is now matrix'''
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


def normpdf(x, loc, scale, logscale, coeff):

    u = (x - loc) / scale

    y = coeff - logscale - (u * u / 2)

    return y


def log_likelihood_3d(model, theta, t, p, s, y, yerr):

    # Extract parameter values
    mu_t, sig_t, mu_p, sig_p, mu_s, sig_s, rho_tp, rho_ts, rho_ps = theta

    # Define model
    model = gauss3d(
        model.norm, t, mu_t, sig_t, p, mu_p, sig_p,
        s, mu_s, sig_s, rho_tp, rho_ts, rho_ps)

    sigma2 = yerr ** 2

    return -0.5 * np.sum((y - model) ** 2 / sigma2)


def log_prior_3d(model, theta):

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
        mut_lnprob = normpdf(
            mu_t,
            loc=model.initial_guess[0],
            scale=model.initial_spread[0],
            logscale=model.log_initial_spread[0],
            coeff=model.norm_coeff)
        sigt_lnprob = normpdf(
            sig_t,
            loc=model.initial_guess[1],
            scale=model.initial_spread[1],
            logscale=model.log_initial_spread[1],
            coeff=model.norm_coeff)
        mup_lnprob = normpdf(
            mu_p,
            loc=model.initial_guess[2],
            scale=model.initial_spread[2],
            logscale=model.log_initial_spread[2],
            coeff=model.norm_coeff)
        sigp_lnprob = normpdf(
            sig_p,
            loc=model.initial_guess[3],
            scale=model.initial_spread[3],
            logscale=model.log_initial_spread[3],
            coeff=model.norm_coeff)
        mus_lnprob = normpdf(
            mu_s,
            loc=model.initial_guess[4],
            scale=model.initial_spread[4],
            logscale=model.log_initial_spread[4],
            coeff=model.norm_coeff)
        sigs_lnprob = normpdf(
            sig_s,
            loc=model.initial_guess[5],
            scale=model.initial_spread[5],
            logscale=model.log_initial_spread[5],
            coeff=model.norm_coeff)
        rhotp_lnprob = normpdf(
            rho_tp,
            loc=model.initial_guess[6],
            scale=model.initial_spread[6],
            logscale=model.log_initial_spread[6],
            coeff=model.norm_coeff)
        rhots_lnprob = normpdf(
            rho_ts,
            loc=model.initial_guess[7],
            scale=model.initial_spread[7],
            logscale=model.log_initial_spread[7],
            coeff=model.norm_coeff)
        rhops_lnprob = normpdf(
            rho_ps,
            loc=model.initial_guess[8],
            scale=model.initial_spread[8],
            logscale=model.log_initial_spread[8],
            coeff=model.norm_coeff)

        return mut_lnprob + sigt_lnprob + mup_lnprob \
            + sigp_lnprob + mus_lnprob + sigs_lnprob \
            + rhotp_lnprob + rhots_lnprob + rhops_lnprob
        # return 0
    else:
        return -np.inf


def log_probability_3d(model, theta, t, p, s, y, yerr):
    lp = log_prior_3d(model, theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_3d(model, theta, t, p, s, y, yerr)
