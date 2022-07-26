"""
Nicholas Hanoian
04/10/2020
generate_figures.py

Generates figures using data processed by preprocessing.py.
"""

import os
import argparse

import pandas as pd

from matplotlib import rc

from plotting.fig_1_success_by_size import fig_1
from plotting.fig_2_team_focus import fig_2
from plotting.fig_3_diversity import fig_3
from plotting.fig_4_leads import fig_4


def get_args():
    """get command line arguments for input_dir, output_dir, start_date, and end_date"""

    # default folder containing data
    # default_input_dir = "/home/nick/downloads/gharchive"
    default_input_dir = "/home/nick/apps/data/github-burstiness/04-22"
    default_output_dir = "figures"

    parser = argparse.ArgumentParser(
        description="Given data processedy by preprocessing.py, generate the figures from the original paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input and output directory args
    parser.add_argument("-i", "--input-dir", help="Input directory",
                        default=default_input_dir, dest="input_dir")
    parser.add_argument("-o", "--output-dir", help="Output directory",
                        default=default_output_dir, dest="output_dir")

    return parser.parse_args()


def main():
    args = get_args()

    repository_stats = pd.read_csv(os.path.join(args.input_dir, "repository_stats_expanded.csv"))
    work_per_member = pd.read_csv(os.path.join(args.input_dir, "work_per_member.csv"))

    # make output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # use Times font and tex rendering
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)

    print("Generating figure 1...")
    fig_1(repository_stats,
          os.path.join(args.output_dir, "1-success-by-team-size.pdf"))

    print("Generating figure 2...")
    fig_2(repository_stats, work_per_member,
          os.path.join(args.output_dir, "2-team-focus.pdf"))

    print("Generating figure 3...")
    fig_3(repository_stats,
          os.path.join(args.output_dir, "3-diversity.pdf"))

    print("Generating figure 4...")
    fig_4(repository_stats,
          os.path.join(args.output_dir, "4-leads.pdf"))


if __name__ == "__main__":
    main()
