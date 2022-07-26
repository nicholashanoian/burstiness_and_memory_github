"""
Nicholas Hanoian
04/10/2020
fig_1_success_by_size.py

Generate figure 1 from the original paper
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def get_fig_1_data(repository_stats):
    """Calculate average success by team size. Filter out top 1% of teams
    for calculation of mean and errors, and calcualte median with original
    data"""

    # filter out those above 99th percentile
    cutoff = repository_stats.max_stargazers.quantile(0.99)
    valid_repos = repository_stats[repository_stats.max_stargazers < cutoff].repo_id

    # only use this for means, use original for medians
    filtered_repository_stats = repository_stats[repository_stats["repo_id"].isin(valid_repos)]

    # calculate summary stats to plot
    means = filtered_repository_stats[["team_size", "max_stargazers"]] \
        .groupby("team_size").mean()[:10]
    margins = filtered_repository_stats[["team_size", "max_stargazers"]] \
        .groupby("team_size").sem()[:10] * 1.96
    medians = repository_stats[["team_size", "max_stargazers"]] \
        .groupby("team_size").median()[:10]

    return means, margins, medians


def fig_1(repository_stats, filename):
    """Mean success by team size with inset for median. Note that it
    appears that the mean plot excludes the top percentile of teams by
    max_stargazers. This is not listed in the paper, but results are much
    too high otherwise, and fairly close with"""
    good_blue = "#4250a2"

    means, margins, medians = get_fig_1_data(repository_stats)

    # plot means
    fig, ax = plt.subplots(figsize=(6.4, 6.4/1.22))
    errs = ax.errorbar(means.index, means["max_stargazers"],
                       yerr=margins["max_stargazers"], capsize=5, lw=2,
                       capthick=2, fmt="o-", zorder=100, clip_on=False, c=good_blue)

    # let points leave the bounds of the axes
    for bound in errs[1]:
        bound.set_clip_on(False)
    for bound in errs[2]:
        bound.set_clip_on(False)

    # label axes and change limits
    ax.set_xlabel("team size, $M$")
    ax.set_ylabel("success, $S$")
    ax.set_xlim(1, 10)
    ax.set_ylim(20, 80)

    # create inset
    axins = inset_axes(ax, width="40%", height="45%", loc=2,
                       bbox_to_anchor=(0.075, -0.025, 1, 1),  # left, bottom, width, height
                       bbox_transform=ax.transAxes)

    # plot medians
    axins.plot(medians.index, medians["max_stargazers"], "o-", c=good_blue,
               clip_on=False, zorder=100)

    # labels and bounds
    axins.set_xlabel("$M$")
    axins.set_ylabel("$S$ (median)")
    axins.set_xlim(1, 10)
    axins.set_ylim(2, 9)
    # specify tick locations
    axins.set_xticks(np.arange(1, 11))
    axins.set_yticks(np.arange(2, 10))

    # save and close
    fig.savefig(filename)
    plt.close(fig)
