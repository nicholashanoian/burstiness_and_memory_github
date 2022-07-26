"""
Nicholas Hanoian
04/10/2020
fig_2_team_focus.py

Generate figure 2 from the original paper
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def fig_2a(work_per_member, ax=None):
    """Fraction of workload by rank. Inset for fraction of work done by
    top ranked member of each team."""

    loosely_dashed = (0, (5, 10))
    # plot main figure
    if ax is None:
        alone = True
        fig, ax = plt.subplots(figsize=(6.4, 6.4/1.15))
    else:
        alone = False

    # iterate through each size of team and compute mean and error for
    # each team member rank
    for i in range(1, 11):
        df = work_per_member[work_per_member["team_size"] == i]
        top_rank = i+1

        frac_work_mean = df.groupby("work_rank")["work_frac"].mean()[:10]
        frac_work_margin = df.groupby("work_rank")["work_frac"].sem()[:10] * 1.96

        ax.errorbar(np.arange(1, top_rank), frac_work_mean,
                    yerr=frac_work_margin, capsize=6, capthick=2, lw=2)

    # labels and limits
    ax.set_yscale("log")
    ax.set_ylabel("fraction of workload, $w_r/W$")
    ax.set_xlabel("rank, $r$")

    ax.set_xlim(1, 10)
    ax.set_ylim(0.01, 1)

    # inset ==============================
    uniform_work = [1/M for M in range(2, 11)]

    # create inset
    axins = inset_axes(ax, width="45%", height="45%", loc=1,
                       bbox_to_anchor=(-0.03, -0.05, 1, 1),  # left, bottom, width, height
                       bbox_transform=ax.transAxes)

    top_workers = work_per_member[work_per_member["work_rank"] == 1]
    means = top_workers[["team_size", "work_frac"]] \
        .groupby("team_size").mean().iloc[1:10]
    errors = top_workers[["team_size", "work_frac"]] \
        .groupby("team_size").sem().iloc[1:10] * 1.96

    axins.plot(range(2, 11), uniform_work, linestyle=loosely_dashed, c="k")
    axins.errorbar(range(2, 11), means["work_frac"],
                   yerr=errors["work_frac"], capsize=5, capthick=2, lw=2, color="k")
    # print(means["work_frac"], errors["work_frac"])

    axins.set_xlim(2, 10)
    axins.set_ylim(0, 1)
    axins.set_ylabel("$w_1/W$")
    axins.set_xlabel("$M$")

    if alone:
        fig.savefig("figures/2a-fraction-workload-by-rank.pdf")
        plt.close(fig)


def get_fig_2b_data(repository_stats, work_per_member):
    """Compute the dominated statistic, and then group into top, average,
    and bottom teams. Then calculate percentage dominated within each
    group as well as corresponding error"""

    # computed whether each repo is dominated or not
    top_workers = work_per_member[work_per_member["work_rank"] == 1]

    repository_stats = repository_stats.merge(top_workers[["repo_id", "work_frac"]],
                                              left_on="repo_id", right_on="repo_id")

    repository_stats["dominated"] = repository_stats["work_frac"] > 0.5

    # creating bounds for groupings of stargazers
    top_bound = repository_stats["max_stargazers"].quantile(0.9)

    average_low = repository_stats["max_stargazers"].quantile(0.4)
    average_high = repository_stats["max_stargazers"].quantile(0.6)

    bottom_bound = repository_stats["max_stargazers"].quantile(0.1)

    # create groups using bounds
    top_stats = repository_stats[repository_stats["max_stargazers"] >= top_bound]
    average_stats = repository_stats[(repository_stats["max_stargazers"] >= average_low) &
                                     (repository_stats["max_stargazers"] <= average_high)]
    bottom_stats = repository_stats[repository_stats["max_stargazers"] <= bottom_bound]

    # percentage and error for top
    top_perc = top_stats[["team_size", "dominated"]] \
        .groupby("team_size").mean()["dominated"].iloc[1:10] * 100
    top_err = top_stats[["team_size", "dominated"]] \
        .groupby("team_size").sem()["dominated"].iloc[1:10] * 1.96 * 100

    # percentage and error for avg
    average_perc = average_stats[["team_size", "dominated"]] \
        .groupby("team_size").mean()["dominated"].iloc[1:10] * 100
    average_err = average_stats[["team_size", "dominated"]] \
        .groupby("team_size").sem()["dominated"].iloc[1:10] * 1.96 * 100

    # percentage and error for bottom
    bottom_perc = bottom_stats[["team_size", "dominated"]] \
        .groupby("team_size").mean()["dominated"].iloc[1:10] * 100
    bottom_err = bottom_stats[["team_size", "dominated"]] \
        .groupby("team_size").sem()["dominated"].iloc[1:10] * 1.96 * 100

    return top_perc, top_err, average_perc, average_err, bottom_perc, bottom_err


def fig_2b(top_perc, top_err, average_perc, average_err, bottom_perc, bottom_err, ax=None):
    """Horizontal dodged bar plot of percentage of dominated teams grouped
    by bottom, avg, and top teams"""
    green = "#70bf48"
    blue = "#4250a2"
    red = "#e41d2c"

    if ax is None:
        alone = True
        fig, ax = plt.subplots(figsize=(6.4, 6.4/1.15))
    else:
        alone = False

    indices = np.arange(2, 11)
    height = np.min(np.diff(indices)) / 4

    ax.barh(indices-height, bottom_perc, height=height, xerr=bottom_err,
            color=green, ecolor=green, alpha=0.5, label="bottom teams")
    ax.barh(indices, average_perc, height=height, xerr=average_err,
            color=blue, ecolor=blue, alpha=0.5, label="average teams")
    ax.barh(indices+height, top_perc, height=height, xerr=top_err,
            color=red, ecolor=red, alpha=0.5, label="top teams")

    ax.set_ylim(10 + 0.5, 2 - 0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("dominated teams (\%)")
    ax.set_ylabel("$M$")

    ax.legend(frameon=False)

    if alone:
        fig.savefig("figures/2b-dominated-teams.pdf")
        plt.close(fig)


def get_fig_2c_data(repository_stats):
    """Split repository_stats into bottom, avg, and top, and generate
    means and errors for effective size by team size"""
    top_bound = repository_stats["max_stargazers"].quantile(0.9)

    average_low = repository_stats["max_stargazers"].quantile(0.4)
    average_high = repository_stats["max_stargazers"].quantile(0.6)

    bottom_bound = repository_stats["max_stargazers"].quantile(0.1)

    top_stats = repository_stats[repository_stats["max_stargazers"] >= top_bound]
    average_stats = repository_stats[(repository_stats["max_stargazers"] >= average_low) &
                                     (repository_stats["max_stargazers"] <= average_high)]
    bottom_stats = repository_stats[repository_stats["max_stargazers"] <= bottom_bound]

    top_means = top_stats[["effective_size", "team_size"]] \
        .groupby("team_size").mean()["effective_size"][:10]
    top_err = top_stats[["effective_size", "team_size"]] \
        .groupby("team_size").sem()["effective_size"][:10] * 1.96

    average_means = average_stats[["effective_size", "team_size"]] \
        .groupby("team_size").mean()["effective_size"][:10]
    average_err = average_stats[["effective_size", "team_size"]] \
        .groupby("team_size").sem()["effective_size"][:10] * 1.96

    bottom_means = bottom_stats[["effective_size", "team_size"]] \
        .groupby("team_size").mean()["effective_size"][:10]
    bottom_err = bottom_stats[["effective_size", "team_size"]] \
        .groupby("team_size").sem()["effective_size"][:10] * 1.96

    return top_means, top_err, average_means, average_err, bottom_means, bottom_err


def fig_2c(top_means, top_err, average_means, average_err, bottom_means, bottom_err, ax=None):
    """Create plot of effective team size by team size"""
    green = "#70bf48"
    blue = "#4250a2"
    red = "#e41d2c"
    loosely_dashed = (0, (5, 10))

    if ax is None:
        alone = True
        fig, ax = plt.subplots(figsize=(6.4, 6.4/1.15))
    else:
        alone = False

    indices = np.arange(1, 11)

    # plot each line
    ax.errorbar(indices, bottom_means, yerr=bottom_err,
                color=green, ecolor=green, alpha=0.6, label="bottom teams",
                capsize=5, lw=2, capthick=2)
    ax.errorbar(indices, average_means, yerr=average_err,
                color=blue, ecolor=blue, alpha=0.6, label="average teams",
                capsize=5, lw=2, capthick=2)
    ax.errorbar(indices, top_means, yerr=top_err,
                color=red, ecolor=red, alpha=0.6, label="top teams",
                capsize=5, lw=2, capthick=2)

    # y=x line
    ax.plot(indices, indices, linestyle=loosely_dashed, color="k")

    # labels and limits
    ax.set_ylim(1, 7)
    ax.set_xlim(1, 10)
    ax.set_xlabel("$M$")
    ax.set_ylabel("effective team size, $m$")
    ax.legend(frameon=False)

    if alone:
        fig.savefig("figures/2c-effective-size.pdf")
        plt.close(fig)


def fig_2d(repository_stats, ax=None):
    """Plot success by m/M for various levels of M. Unsure how the binning
    in the original paper was done. We assume that we want equal-width
    bins ranging from 1/M to 1. It appears that there are a different
    number of bins for various levels of M in the original paper, but
    we just kept this constant here. We are not sure of the rationale
    for using a different number of bins for M>=8 as for M=6. Points
    are plotted in the middle of the bin.

    """

    n_bins = 7

    if ax is None:
        alone = True
        fig, ax = plt.subplots(figsize=(6.4, 6.4/1.15))
    else:
        alone = False

    repository_stats["ratio"] = repository_stats["effective_size"] / repository_stats["team_size"]

    colors = ["#42439b", "#4e5eaa", "#5ab6e7", "#8fd0b5", "#cade6b", "#ffcb3f", "#f26443"]

    def plot_group(df, bins, label, color, just_means=False):
        """Plot errorbar line for one group of M=M"""
        means = []
        sems = []
        x_vals = []
        # iterate through the bins
        for i in range(len(bins) - 1):
            x_vals.append((bins[i] + bins[i+1]) / 2) # middle of bin
            # get rows within this bin
            one_bin = df[(df["ratio"] >= bins[i]) & (df["ratio"] < bins[i+1])]
            # calculate stats
            means.append(one_bin["max_stargazers"].mean())
            sems.append(one_bin["max_stargazers"].sem() * 1.96)

        # plot the line
        if just_means: # for all line
            ax.plot(x_vals, means, label=label, color=color, lw=4, zorder=100)
        else: # for everything else
            ax.errorbar(x_vals, means, yerr=sems, label=label, color=color, alpha=0.9,
                        capsize=5, lw=2, capthick=2)

    # M from 2 to 7
    for M, color in zip(range(2, 8), colors):
        df = repository_stats[repository_stats["team_size"] == M]
        bins = np.linspace(1/M, 1, n_bins)
        plot_group(df, bins, f"$M={M}$", color)

    # M > 8
    df = repository_stats[(repository_stats["team_size"] >= 8) &
                          (repository_stats["team_size"] <= 10)]
    bins = np.linspace(1/10, 1, n_bins)
    plot_group(df, bins, f"$M \geq 8$", colors[6])

    # 2 <= M <= 10
    df = repository_stats[(repository_stats["team_size"] >= 2) & (repository_stats["team_size"] <= 10)]
    bins = np.linspace(1/10, 1, 15)
    plot_group(repository_stats, bins, "all", "k", just_means=True)

    # labels and limits
    ax.legend(frameon=False)

    ax.set_ylim(0, 1600)
    ax.set_xlim(0, 1)
    ax.set_ylabel("$S$")
    ax.set_xlabel("$m/M$")

    # save figure
    if alone:
        fig.savefig("figures/2d-ratio.pdf")
        plt.close(fig)


def fig_2(repository_stats, work_per_member, filename):
    """Plot all 4 subplots of figure 2 together"""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.33))
    # fig, axs = plt.subplots(2, 2, figsize=(6.5, 5))

    axs = axs.flatten()

    fig_2a(work_per_member, ax=axs[0])

    fig_2b_data = get_fig_2b_data(repository_stats, work_per_member)
    fig_2b(*fig_2b_data, ax=axs[1])

    fig_2c_data = get_fig_2c_data(repository_stats)
    fig_2c(*fig_2c_data, ax=axs[2])

    fig_2d(repository_stats, ax=axs[3])

    # subplot labels
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for i, ax in enumerate(axs):
        ax.text(-0.2, 1, "\\textit{" + letters[i] + "}", transform=ax.transAxes, size=12)

    # add horizontal space between plots
    plt.subplots_adjust(wspace=0.3)
    fig.savefig(filename)
    plt.close(fig)
