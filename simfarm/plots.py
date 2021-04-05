import corner
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from os.path import abspath, dirname, join


PARENT_DIR = dirname(dirname(abspath(__file__)))


def create_all_plots(simfarm):
    plot_validation(simfarm)
    print('validation plot in model_performance')
    plot_walkers(simfarm)
    print('walkers plot in model_performance/Chains')
    plot_response(simfarm)
    print('responses plotted in Response Functions')
    post_prior_comp(simfarm)
    print('post_prior_comp plotted in model_performance/Corners')
    true_pred_comp(simfarm)
    print('true_pred_comp plotted model_performance/Predictionvstruth')
    climate_dependence(simfarm)
    print('climate_dependence plotted in Climate Analysis')



def plot_validation(simfarm):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(np.arange(simfarm.preds.size), simfarm.resi, marker="+", label="Regions")
    ax.axhline(np.mean(simfarm.resi), linestyle="-", color="k", label="Mean")
    ax.axhline(np.median(simfarm.resi), linestyle="--", color="k", label="Median")

    ax.set_xlabel("Region")
    ax.set_ylabel("$1 - Y_{\mathrm{Pred}} / Y_{\mathrm{True}}$ (%)")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.savefig(
        join(
            PARENT_DIR, "model_performance",
            f"Validation{simfarm.cult}_3d.png"),
        bbox_inches="tight")


def plot_walkers(simfarm):

    ndim = len(simfarm.initial_guess)

    for i in range(ndim):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(simfarm.model.chain[:, :, i].T, "-", color="k",
                        alpha=0.3)
        fig.savefig(
            join(
                PARENT_DIR,
                "model_performance", "Chains",
                f"samplerchain_{i}_{simfarm.cult}_daily_3d.png"),
            bbox_inches="tight")
        plt.close(fig)


def plot_response(simfarm):

    # Create arrays to evaluate response function at
    eval_t = np.linspace(0, 6000, 1000)
    eval_p = np.linspace(0, 3500, 1000)
    eval_s = np.linspace(0, 3500, 1000)

    # Get grid of values
    tp_tt, tp_pp = np.meshgrid(eval_t, eval_p)
    ts_tt, ts_ss = np.meshgrid(eval_t, eval_s)
    ps_pp, ps_ss = np.meshgrid(eval_p, eval_s)

    # Compute temperature response
    t_resp = simfarm.gauss3d(simfarm.norm, eval_t, simfarm.mean_params["mu_t"],
                            simfarm.mean_params["sig_t"], 0,
                            simfarm.mean_params["mu_p"],
                            simfarm.mean_params["sig_p"], 0,
                            simfarm.mean_params["mu_s"],
                            simfarm.mean_params["sig_s"],
                            simfarm.mean_params["rho_tp"],
                            simfarm.mean_params["rho_ts"],
                            simfarm.mean_params["rho_ps"])

    # Compute precipitation response
    p_resp = simfarm.gauss3d(simfarm.norm, 0, simfarm.mean_params["mu_t"],
                            simfarm.mean_params["sig_t"], eval_p,
                            simfarm.mean_params["mu_p"],
                            simfarm.mean_params["sig_p"], 0,
                            simfarm.mean_params["mu_s"],
                            simfarm.mean_params["sig_s"],
                            simfarm.mean_params["rho_tp"],
                            simfarm.mean_params["rho_ts"],
                            simfarm.mean_params["rho_ps"])

    # Compute sunshine response
    s_resp = simfarm.gauss3d(simfarm.norm, 0, simfarm.mean_params["mu_t"],
                            simfarm.mean_params["sig_t"], 0,
                            simfarm.mean_params["mu_p"],
                            simfarm.mean_params["sig_p"], eval_s,
                            simfarm.mean_params["mu_s"],
                            simfarm.mean_params["sig_s"],
                            simfarm.mean_params["rho_tp"],
                            simfarm.mean_params["rho_ts"],
                            simfarm.mean_params["rho_ps"])

    # Compute the response grids
    resp_grid_tp = simfarm.gauss2d_resp(tp_tt, simfarm.norm,
                                        simfarm.mean_params["mu_t"],
                                        simfarm.mean_params["sig_t"], tp_pp,
                                        simfarm.mean_params["mu_p"],
                                        simfarm.mean_params["sig_p"],
                                        simfarm.mean_params["rho_tp"])
    resp_grid_ts = simfarm.gauss2d_resp(ts_tt, simfarm.norm,
                                        simfarm.mean_params["mu_t"],
                                        simfarm.mean_params["sig_t"], ts_ss,
                                        simfarm.mean_params["mu_s"],
                                        simfarm.mean_params["sig_s"],
                                        simfarm.mean_params["rho_ts"])
    resp_grid_ps = simfarm.gauss2d_resp(ps_pp, simfarm.norm,
                                        simfarm.mean_params["mu_p"],
                                        simfarm.mean_params["sig_p"], ps_ss,
                                        simfarm.mean_params["mu_s"],
                                        simfarm.mean_params["sig_s"],
                                        simfarm.mean_params["rho_ps"])

    # Set up figure
    fig = plt.figure(figsize=(16, 12))
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
    cbar1.ax.set_xlabel(
        simfarm.metric + " (" + simfarm.metric_units + "month$^{-1}$)",
        fontsize=10, color="k", labelpad=5)
    cbar1.ax.xaxis.set_label_position("top")
    cbar1.ax.tick_params(axis="x", labelsize=10, color="k", labelcolor="k")
    cbar2.ax.set_xlabel(
        simfarm.metric + " (" + simfarm.metric_units + "month$^{-1}$)",
        fontsize=10, color="k", labelpad=5)
    cbar2.ax.xaxis.set_label_position("top")
    cbar2.ax.tick_params(axis="x", labelsize=10, color="k", labelcolor="k")
    cbar3.ax.set_xlabel(
        simfarm.metric + " (" + simfarm.metric_units + "month$^{-1}$)",
        fontsize=10, color="k", labelpad=5)
    cbar3.ax.xaxis.set_label_position("top")
    cbar3.ax.tick_params(axis="x", labelsize=10, color="k", labelcolor="k")

    ax4.plot(eval_t, t_resp)
    ax5.plot(eval_p, p_resp)
    ax6.plot(eval_s, s_resp)

    # Label axes
    ax1.set_xlabel(r"GDD ($^\circ$C days)")
    ax1.set_ylabel(r"$\sum P$ (mm)")
    ax2.set_xlabel(r"GDD ($^\circ$C days)")
    ax2.set_ylabel(r"$\sum S$ (hrs)")
    ax3.set_xlabel(r"$\sum P$ (mm)")
    ax3.set_ylabel(r"$\sum S$ (hrs)")
    ax4.set_xlabel(r"GDD ($^\circ$C days)")
    ax4.set_ylabel(
        simfarm.metric + " (" + simfarm.metric_units + "month$^{-1}$)")
    ax5.set_xlabel(r"$\sum P$ (mm)")
    ax5.set_ylabel(
        simfarm.metric + " (" + simfarm.metric_units + "month$^{-1}$)")
    ax6.set_xlabel(r"$\sum S$ (hrs)")
    ax6.set_ylabel(
        simfarm.metric + " (" + simfarm.metric_units + "month$^{-1}$)")

    # Save the figure
    fig.savefig(
        join(
            PARENT_DIR,
            "Response_functions", 
            f"response_{simfarm.cult}_daily_3d.png"),
        dpi=300, bbox_inches="tight")


def post_prior_comp(simfarm):

    labels = [r"$\mu_t$", r"$\sigma_t$", r"$\mu_p$", r"$\sigma_p$",
                r"$\mu_s$", r"$\sigma_s$",
                r"$\rho_tp$", r"$\rho_ts$", r"$\rho_ps$"]
    fig = corner.corner(simfarm.flat_samples, show_titles=True, labels=labels,
                        plot_datapoints=True,
                        quantiles=[0.16, 0.5, 0.84])

    fig.savefig(
        join(
            PARENT_DIR,
            "model_performance", "Corners", 
            f"corner_{simfarm.cult}_daily_3d.png"),
        bbox_inches="tight")

    plt.close(fig)


def true_pred_comp(simfarm):

    # Set up figure
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)

    im = ax1.scatter(simfarm.predict_yields, simfarm.preds, marker="+")
    ax1.plot((7, 15), (7, 15), linestyle="--", color="k", label="1-1")

    # Label axes
    ax1.set_xlabel(r"True Yeild (t Ha$^{-1}$)")
    ax1.set_ylabel(r"Predicted Yeild (t Ha$^{-1}$)")

    ax1.legend()

    # Save the figure
    fig.savefig(
        join(
            PARENT_DIR,
            "model_performance", "Predictionvstruth",
            f"prediction_vs_truth_{simfarm.cult}.png"),
        bbox_inches="tight")

def climate_dependence(simfarm):

    # Set up figure
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.scatter(simfarm.therm_days, simfarm.yield_data, marker="+")
    ax2.scatter(simfarm.tot_precip, simfarm.yield_data, marker="+")
    ax3.scatter(simfarm.tot_sun, simfarm.yield_data, marker="+")

    strt_line = lambda x, m, c: m * x + c

    popt, pcov = curve_fit(strt_line, simfarm.therm_days, simfarm.yield_data,
                            p0=(1, 4))

    xs = np.linspace(np.min(simfarm.therm_days),
                        np.max(simfarm.therm_days),
                        1000)

    ax1.plot(xs, strt_line(xs, popt[0], popt[1]),
                linestyle="--", color="k",
                label="Fit: $y = $%.4f" % popt[0]
                    + "$\mathrm{GDD} +$ %.2f" % popt[1])

    popt, pcov = curve_fit(strt_line, simfarm.tot_precip, simfarm.yield_data,
                            p0=(1, 4))

    xs = np.linspace(np.min(simfarm.tot_precip),
                        np.max(simfarm.tot_precip),
                        1000)

    ax2.plot(xs, strt_line(xs, popt[0], popt[1]),
                linestyle="--", color="k",
                label="Fit: $y = $%.4f" % popt[0]
                    + r"$\times(\sum P)+$%.2f" % popt[1])

    popt, pcov = curve_fit(strt_line, simfarm.tot_sun, simfarm.yield_data,
                            p0=(1, 4))

    xs = np.linspace(np.min(simfarm.tot_sun),
                        np.max(simfarm.tot_sun),
                        1000)

    ax3.plot(xs, strt_line(xs, popt[0], popt[1]),
                linestyle="--", color="k",
                label="Fit: $y = $%.4f" % popt[0]
                    + r"$\times(\sum S)+$%.2f" % popt[1])

    # Label axes
    ax1.set_ylabel(r"Yeild (t Ha$^{-1}$)")
    ax1.set_xlabel(r"GDD ($^\circ$C days)")
    ax2.set_xlabel(r"$\sum P$ (mm)")
    ax3.set_xlabel(r"$\sum S$ (hrs)")

    for ax in [ax2, ax3]:
        ax.tick_params(axis='y', left=False, right=False, labelleft=False,
                        labelright=False)

        ax.legend()

    # Save the figure
    fig.savefig(
        join(
            PARENT_DIR, "Climate_analysis",
            f"input_yield_climate_{simfarm.cult}1d.png"),
        bbox_inches="tight")
