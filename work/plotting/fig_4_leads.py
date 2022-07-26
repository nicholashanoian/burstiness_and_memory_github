"""
Nicholas Hanoian
04/10/2020
fig_4_leads.py

Generate figure 4 from the original paper
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import numpy as np





def fig_4a(repository_stats, fig=None, ax=None):
    """3D bar chart of success by team size and leads. Very finnicky and
    needed to do some hack-y things to get it to look right"""

    # filter out those above 99th percentile
    cutoff = repository_stats.max_stargazers.quantile(0.99)
    valid_repos = repository_stats[repository_stats.max_stargazers < cutoff].repo_id

    # only use this for means, use original for medians
    repository_stats = repository_stats[repository_stats["repo_id"].isin(valid_repos)]

    means = repository_stats[(repository_stats["team_size"] <= 10) &
                             (repository_stats["n_leads"] <= 10)] \
        .groupby(["team_size", "n_leads"]).mean()["max_stargazers"] \
        .reset_index().sort_values(by="n_leads").sort_values(by="n_leads")

    # create figure if needed
    if ax is None:
        alone = True
        fig = plt.figure(figsize=(6.4, 6.4/1.18))
        ax = fig.add_subplot(projection="3d")
    else:
        alone = False

    # set up viewing angle
    ax.view_init(25, -35)

    # min and max for color map
    color_min = 0
    color_max = 320

    # get colors for bars
    norm = colors.Normalize(color_min, color_max)
    color_values = cm.jet(norm(means["max_stargazers"].tolist()))

    # offset x and y by 0.25 so that they are centered on the ticks
    # x,y,z are the position of the base of the bars dx,dy,dz are the
    # width,length, and height some overlap is messy, and zsort="max"
    # looks a little better than zsort="average". Possible solution
    # for zsort problem: https://stackoverflow.com/a/37374864
    ax.bar3d(x=means["team_size"] - 0.25, y=means["n_leads"] - 0.25, z=0,
             dx=0.5, dy=0.5, dz=means["max_stargazers"],
             zsort="max", color=color_values, edgecolor="k", linewidth=0.5)

    # colorbar -- changed to look good with both subplots, so won't look good alone any more
    #                   left, bottom, width, height
    ax1 = fig.add_axes([0.05, 0.75, 0.25, 0.025])
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cm.jet, norm=norm,
                                    orientation='horizontal')
    # label and ticks for colorbar
    cb1.set_label("success, $S$")
    cb1.set_ticks(np.arange(color_min, color_max, 40))

    # axis limits
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_zlim(0, 350)

    # axis ticks
    ax.set_xticks(np.arange(1, 11))
    ax.set_yticks(np.arange(1, 11))

    # axis labels
    ax.set_xlabel("team size, $M$")
    ax.set_ylabel("no. leads, $L$")
    ax.set_zlabel("success, $S$")

    # removed colored backgruond from walls
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # make grid lighter
    grid_style = {'grid': {'color': "#e6e6e6", 'linewidth': 0.8, 'linestyle': '-'}}
    ax.w_xaxis._axinfo.update(grid_style)
    ax.w_yaxis._axinfo.update(grid_style)
    ax.w_zaxis._axinfo.update(grid_style)

    # "zoom in" on the view to make the plot bigger in joint figure 4
    ax.dist = 8

    # remove padding between the tick labels and the axis
    ax.tick_params(axis='both', which='major', pad=-3)

    # remove padding between the axis labels and the axis
    ax.xaxis.labelpad = -7
    ax.yaxis.labelpad = -7
    ax.zaxis.labelpad = -7

    # save figure -- won't look good any more because of changes made for joint figure
    if alone:
        fig.savefig("figures/4a-3d-leads.pdf")
        plt.close(fig)


def fig_4b(repository_stats, ax=None):
    """Mean success by n_leads, grouped by team size"""

    colors = ["#00a4ff", "#40ffb7", "#b7ff40", "#ffb900", "#ff3000", "#800000"]
    markers = ["d", "H", "h", "^", "s", "o"]

    if ax is None:
        alone = True
        fig, ax = plt.subplots(figsize=(6.4, 6.4/1.18))
    else:
        alone = False

    def plot_group(df, M, label, color, marker):
        """Plot line for one value of M"""
        x_vals = np.arange(1, M+1)
        means = df[["n_leads", "max_stargazers"]].groupby("n_leads").mean()[:10]
        sems = df[["n_leads", "max_stargazers"]].groupby("n_leads").sem()[:10] * 1.96

        errs = ax.errorbar(x_vals, means["max_stargazers"], yerr=sems["max_stargazers"],
                           label=label, color=color, marker=marker,
                           alpha=0.9, capsize=5, lw=2, capthick=2, zorder=100, clip_on=False,)

        # let points leave the bounds of the axes
        for bound in errs[1]:
            bound.set_clip_on(False)
        for bound in errs[2]:
            bound.set_clip_on(False)

    # filter out those above 99th percentile
    cutoff = repository_stats.max_stargazers.quantile(0.99)
    valid_repos = repository_stats[repository_stats.max_stargazers < cutoff].repo_id

    # only use this for means, use original for medians
    repository_stats = repository_stats[repository_stats["repo_id"].isin(valid_repos)]

    # M = 2,3,4,5,6
    for M, color, marker in zip([2, 3, 4, 5, 6], colors, markers):
        df = repository_stats[repository_stats["team_size"] == M]
        plot_group(df, M, f"$M={M}$", color, marker)

    # M >= 7
    df = repository_stats[repository_stats["team_size"] >= 7]
    plot_group(df, 10, "$M \\geq 7$", colors[-1], markers[-1])

    # label axes and change limits
    ax.set_xlabel("no. leads, $L$")
    ax.set_ylabel("$S$")
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 200)

    ax.legend(frameon=False)

    # save and close
    if alone:
        fig.savefig("figures/4-success-by-team-size.pdf")
        plt.close(fig)


def fig_4(repository_stats, filename):
    """Figure 4 from the original paper about team leads. 3D plot is
    finnicky, and needed to do hack-y stuff to get it to look right"""
    fig = plt.figure(figsize=(9, 3.5))

    # make the 3d plot take up a little more space in the plot
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[15, 14])

    # set up axes
    ax0 = fig.add_subplot(gs[0], projection="3d")
    ax1 = fig.add_subplot(gs[1])

    # plot on the axes
    fig_4a(repository_stats, fig, ax0)
    fig_4b(repository_stats, ax1)

    # adjust margins of subplots. Move very close to left edge because
    # of whitespace in 3d plot
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, wspace=0.35)
    fig.savefig(filename)
    plt.close(fig)
