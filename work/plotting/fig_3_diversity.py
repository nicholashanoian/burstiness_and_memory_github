"""
Nicholas Hanoian
04/10/2020
fig_3_diversity.py

Generate figure 3 from the original paper
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
from scipy.stats import spearmanr


def fig_3(repository_stats, filename):
    """Plot success by diversity, grouped by team size. Just as with 2d,
    binning is not currently quite right"""

    # filter out those above 99th percentile
    cutoff = repository_stats.max_stargazers.quantile(0.99)
    valid_repos = repository_stats[repository_stats.max_stargazers < cutoff].repo_id
    repository_stats = repository_stats[repository_stats["repo_id"].isin(valid_repos)]

    fig, ax = plt.subplots(figsize=(6, 6/1.178))

    # styles to be used later
    colors = ["#440154", "#3b528b", "#21918c", "#5ec962"]
    markers = ["o", "s", "d", "^"]
    loosely_dashed = (0, (5, 10))

    def plot_group(df, bins, label, color, marker):
        """Plot errorbar line for one group of M=M"""
        means = []
        sems = []
        x_vals = []
        # iterate through the bins
        for i in range(len(bins) - 1):
            x_vals.append((bins[i] + bins[i+1]) / 2)  # middle of bin
            # get rows within this bin
            one_bin = df[(df["diversity"] >= bins[i]) & (df["diversity"] < bins[i+1])]
            # calculate stats
            means.append(one_bin["max_stargazers"].mean())
            sems.append(one_bin["max_stargazers"].sem() * 1.96)

        # plot the line
        ax.errorbar(x_vals, means, yerr=sems, label=label, color=color, marker=marker,
                    alpha=0.9, capsize=5, lw=2, capthick=2)

    # M = 2,3,4
    for M, color, marker in zip([2, 3, 4], colors, markers):
        df = repository_stats[repository_stats["team_size"] == M]
        bins = np.linspace(1/M, 1, 10)
        plot_group(df, bins, f"$M={M}$", color=color, marker=marker)

    # M >= 5
    df = repository_stats[repository_stats["team_size"] >= 5]
    bins = np.linspace(1/repository_stats["team_size"].max(), 1, 10)
    plot_group(df, bins, "$M \\geq 5$", color=colors[-1], marker=markers[-1])

    # mean stargazers line
    mean_stargazers = [repository_stats["max_stargazers"].mean()] * 2
    ax.plot([0, 1], mean_stargazers, linestyle=loosely_dashed, color="k")

    # limits and labels
    ax.set_xlim(0, 1)
    ax.set_yticks([50, 100, 150, 200])
    ax.set_ylim(0, 200)
    ax.set_xlabel("diversity, $D$")
    ax.set_ylabel("$S$")

    # inset
    team_sizes = np.arange(2, 11)
    corrs = []
    # calculate spearman rank correlation for each team size in interest
    for size in team_sizes:
        df = repository_stats[repository_stats["team_size"] == size]
        rho, pval = spearmanr(df["diversity"], df["max_stargazers"])
        corrs.append(rho)

    # create inset
    axins = inset_axes(ax, width="38%", height="35%", loc=2,
                       bbox_to_anchor=(0.15, -0.025, 1, 1),  # left, bottom, width, height
                       bbox_transform=ax.transAxes)

    # plot inset line
    points = axins.plot(team_sizes, corrs, ".-k")
    for point in points:
        point.set_clip_on(False)

    # limits and labels
    axins.set_xlim(2, 10)
    axins.set_ylim(0, 0.4)
    axins.set_xlabel("$M$")
    axins.set_ylabel("$\\rho(D,S)$")

    ax.legend(frameon=False, loc="lower left",
              bbox_to_anchor=(0.01, 0.15, 1, 1), bbox_transform=ax.transAxes)

    # save plot
    fig.savefig(filename)
    plt.close(fig)
