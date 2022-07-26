"""
Nicholas Hanoian
04/10/2020
preprocessing.py

Given data aggregated from tidy_data.py, generate two additional csvs
repository_stats_expanded.csv and work_per_member.csv which contain
the data needed to generate the figures from the original paper.
"""

import os
import argparse

import pandas as pd
import numpy as np

from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm


# ==========================================================
# Arguments
# ==========================================================

def get_args():
    """get command line arguments for input_dir, output_dir, start_date, and end_date"""

    # default folder containing data
    # default_input_dir = "/home/nick/downloads/gharchive"
    default_input_dir = "/home/nick/apps/data/github-burstiness/04-22"
    default_output_dir = default_input_dir

    parser = argparse.ArgumentParser(
        description="Given data aggregated from tidy_data.py, generate two additional csvs repository_stats_expanded.csv and work_per_member.csv which contain the data needed to generate the figures from the original paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input and output directory args
    parser.add_argument("-i", "--input-dir", help="Input directory",
                        default=default_input_dir, dest="input_dir")
    parser.add_argument("-o", "--output-dir", help="Output directory",
                        default=default_output_dir, dest="output_dir")

    parser.add_argument("-j", "--jobs",
                        help="Number of cores to use for multithreading. Defaults too all cores detected on your machine",
                        default=cpu_count(), dest="n_threads", type=int)

    return parser.parse_args()


# ==========================================================
# Utility Functions
# ==========================================================

def flatten(lst):
    """Turn list of lists into a flat list"""
    return [item for sublist in lst for item in sublist]


def chunkify(l, n_chunks):
    """Split list l into n_chunks chunks of equal size"""
    step_size = int(np.ceil(len(l) / n_chunks))
    for i in range(0, len(l), step_size):
        yield l[i:i + step_size]


# ==========================================================
# Filtering and computing aditional statistics
# ==========================================================

def initial_filter(all_repository_stats, all_team_members, all_pushes, allow_no_stars=False):
    """Perform overall filter of all data. Remove repos with 0 stargazers,
    and those which have less than 2*N_MONTHS commits over the entire
    period"""

    # number of months the data spans
    N_MONTHS = 15

    # filter repos
    # get ids of repos with at least 1 star
    valid_repos_by_stars = set(all_repository_stats[
        all_repository_stats["max_stargazers"] > 0]["repo_id"])

    # get ids of repos with at least 2 commits per month, on average
    push_counts = all_pushes["repo_id"].value_counts()
    valid_repos_by_pushes = set(push_counts[push_counts >= 2*N_MONTHS].keys())

    # intersection of constraints to get set of valid repos
    if allow_no_stars:
        valid_repos = valid_repos_by_pushes
    else:
        valid_repos = valid_repos_by_stars & valid_repos_by_pushes

    # print("total repos:", len(valid_repos), ". Should be 151,542")

    # filtered data
    repository_stats = all_repository_stats[all_repository_stats["repo_id"].isin(valid_repos)]
    pushes = all_pushes[all_pushes["repo_id"].isin(valid_repos)]
    team_members = all_team_members[all_team_members["repo_id"].isin(valid_repos)]

    return repository_stats, pushes, team_members


def add_team_sizes(repository_stats, team_members):
    """Add team sizes to repository_stats"""
    team_sizes = team_members["repo_id"].value_counts()
    team_sizes = team_sizes.reset_index()
    team_sizes.columns = ["repo_id", "team_size"]
    return team_sizes.merge(repository_stats, left_on="repo_id", right_on="repo_id")


def add_work(repository_stats, pushes):
    """Create dataframe of work per member, fraction of work per member,
    and rank of member. Also add total work per team to
    repository_stats"""
    # count number of rows with pairs of repo,user
    work_per_member = pushes.groupby(["repo_id", "user"]).count()

    # drop unneeded columns and rename count column to work
    work_per_member = work_per_member.reset_index()
    work_per_member = work_per_member.drop(columns=["size", "location"])
    work_per_member.columns = ["repo_id", "user", "work"]

    # group by repo to get counts per repo
    work_per_repo = work_per_member.groupby("repo_id").sum()
    work_per_repo = work_per_repo.reset_index()
    work_per_repo.columns = ["repo_id", "total_work"]

    repository_stats = repository_stats.merge(work_per_repo,
                                              left_on="repo_id", right_on="repo_id")

    # compute fraction of workload for each member
    work_per_member = work_per_member.merge(work_per_repo,
                                            left_on="repo_id", right_on="repo_id")
    work_per_member["work_frac"] = work_per_member["work"] / work_per_member["total_work"]
    work_per_member["work_rank"] = work_per_member \
        .groupby("repo_id")["work_frac"].rank(ascending=False, method="first").astype(int)

    # add team size to work_per_member
    work_per_member = work_per_member.merge(repository_stats[["repo_id", "team_size"]],
                                            left_on="repo_id", right_on="repo_id")

    return repository_stats, work_per_member


def add_ages(repository_stats, end_date=pd.to_datetime("2014-04-01 23:59:59")):
    created_ats = pd.to_datetime(repository_stats["created_at"], utc=True)
    created_ats = pd.DatetimeIndex(created_ats).tz_convert(None)

    ages = end_date - created_ats
    # in days
    ages = [age.total_seconds() / (60*60*24) for age in ages]
    repository_stats["age"] = ages
    return repository_stats


def add_effective_team_size(repository_stats, work_per_member):
    """Compute effective team size and add it to the repository_stats dataframe"""
    repo_ids = []
    effective_sizes = []
    for repo_id in tqdm(repository_stats["repo_id"]):
        member_fracs = work_per_member[work_per_member["repo_id"] == repo_id]["work_frac"]
        H = -1 * sum(member_fracs * np.log2(member_fracs))
        repo_ids.append(repo_id)
        effective_sizes.append(2**H)

    effective_df = pd.DataFrame({"repo_id": repo_ids, "effective_size": effective_sizes})

    return repository_stats.merge(effective_df, left_on="repo_id", right_on="repo_id")


# ==========================================================
# Experience and diversity
# ==========================================================

def calc_one_experience_diversity(repo_id, repository_stats, team_members_no_stars):
    """Calculate experience and diversity for one repository"""

    # get list of team members of the repo in interest
    members = team_members_no_stars[team_members_no_stars["repo_id"] == repo_id]["user"]
    team_size = len(members)

    # get list of list of projects of each member on the team
    projects = []
    for member in members:
        projects.append(team_members_no_stars[
            team_members_no_stars["user"] == member]["repo_id"].tolist())

    # total number of projects \sum_i |R_i|
    n_total_projects = sum([len(project) for project in projects])
    # number of unique projects | \cup_i R_i |
    n_unique_projects = len(set(flatten(projects)))

    # calculate experience and diversity
    experience = 1/team_size * (n_total_projects - 1)
    diversity = n_unique_projects / n_total_projects

    return {"repo_id": repo_id, "experience": experience, "diversity": diversity}


def calc_many_experience_diversity(repo_ids, repository_stats, team_members_no_stars):
    """Calculate experience and diversity for a set of repos"""
    results = []
    for repo_id in tqdm(repo_ids):
        results.append(calc_one_experience_diversity(repo_id, repository_stats,
                                                     team_members_no_stars))
    return results


def add_experience_diversity(repository_stats, team_members_no_stars, n_threads):
    """Add experience and diversity to repository_stats. Slow process, so
    parallelized"""

    # note that we shuffle the repo ids because for some reason they
    # are sorted by team size, so if we don't some threads will finish
    # much earlier than others
    results = Parallel(n_jobs=n_threads)(delayed(calc_many_experience_diversity)
                                  (repo_ids, repository_stats, team_members_no_stars)
                                  for repo_ids in chunkify(repository_stats["repo_id"].sample(frac=1), n_threads))

    return repository_stats.merge(pd.DataFrame(flatten(results)),
                                  left_on="repo_id", right_on="repo_id")


# ==========================================================
# Leads
# ==========================================================

def determine_one_lead(user, repository_stats, work_per_member):
    """Determine whether user is a lead on any of their projects"""
    # iterate through all projects this user worked on
    projects = work_per_member[work_per_member["user"] == user]
    for idx, proj in projects.iterrows():
        # amount of work that others did on this project
        others_work = work_per_member[work_per_member["repo_id"] == proj["repo_id"]]["work"]
        # if we are the max amount of work, we're the lead
        if max(others_work) == proj["work"]:
            return {"user": user, "is_lead": True}

    return {"user": user, "is_lead": False}


def determine_many_leads(users, repository_stats, work_per_member):
    """Calculate experience and diversity for a set of repos"""
    results = []
    for user in tqdm(users, desc="Determining leads"):
        results.append(determine_one_lead(user, repository_stats, work_per_member))
    return results


def calc_one_n_leads(repo_id, work_per_member, user_leads):
    """Calcualte the number of leads for one repository"""
    members = work_per_member[work_per_member["repo_id"] == repo_id]
    members = members.merge(user_leads, left_on="user", right_on="user")
    return {"repo_id": repo_id, "n_leads": len(members[members["is_lead"] == True])}


def calc_many_n_leads(repo_ids, work_per_member, user_leads):
    """Calculate the number of leads for several repositories"""
    results = []
    for repo_id in tqdm(repo_ids, desc="Calculating n_leads"):
        results.append(calc_one_n_leads(repo_id, work_per_member, user_leads))
    return results


def add_n_leads(repository_stats, work_per_member, n_threads):
    """Calculate the number of leads for each repository and add it to
    repository_stats"""
    # iterate through all users and determine whether they are leads on any of their projects
    all_users = work_per_member["user"].unique()

    results = Parallel(n_jobs=n_threads)(delayed(determine_many_leads)
                                         (users, repository_stats, work_per_member)
                                         for users in chunkify(all_users, n_threads))

    user_leads = pd.DataFrame(flatten(results))

    # calculate the number of leads of each repository
    results = Parallel(n_jobs=n_threads)(delayed(calc_many_n_leads)
                                         (repo_ids, work_per_member, user_leads)
                                         for repo_ids in chunkify(repository_stats["repo_id"].sample(frac=1), n_threads))

    n_leads = pd.DataFrame(flatten(results))

    return repository_stats.merge(n_leads, left_on="repo_id", right_on="repo_id")


# ==========================================================
# Putting it all together
# ==========================================================

def expand_repository_stats(repository_stats, pushes, team_members, team_members_no_stars, n_threads):
    """Run all computations to expand repository_stats to include the
    additional measures, as well as create work_per_member dataframe."""

    print("Calculating team sizes (1/6)")
    repository_stats = add_team_sizes(repository_stats, team_members)

    print("Calculating ages (2/6)")
    repository_stats = add_ages(repository_stats)
    
    print("Calculating work per team member (3/6)")
    repository_stats, work_per_member = add_work(repository_stats, pushes)

    print("Calculating effective team size (4/6)")
    repository_stats = add_effective_team_size(repository_stats, work_per_member)

    print("Calculating experience and diversity (5/6)")
    repository_stats = add_experience_diversity(repository_stats, team_members_no_stars,
                                                n_threads)

    print("Calculating number of leads (6/6)")
    repository_stats = add_n_leads(repository_stats, work_per_member, n_threads)

    return repository_stats, work_per_member


def main():
    args = get_args()

    # read in data produced by tidy_data.py
    print("Reading in data from", args.input_dir)
    all_repository_stats = pd.read_csv(os.path.join(args.input_dir, "repository_stats.csv"))
    all_pushes = pd.read_csv(os.path.join(args.input_dir, "pushes.csv"))
    all_team_members = pd.read_csv(os.path.join(args.input_dir, "team_members.csv"))

    print("Filtering out insignificant teams")
    repository_stats, pushes, team_members = initial_filter(
        all_repository_stats, all_team_members, all_pushes)

    _, _, team_members_no_stars = initial_filter(
        all_repository_stats, all_team_members, all_pushes, allow_no_stars=True)

    print("Calculating additional statistics")
    repository_stats, work_per_member = expand_repository_stats(
        repository_stats, pushes, team_members,
        team_members_no_stars, args.n_threads)

    print("Writing new csvs to", args.output_dir)
    repository_stats.to_csv(os.path.join(args.output_dir, "repository_stats_expanded.csv"), index=False)
    work_per_member.to_csv(os.path.join(args.output_dir, "work_per_member.csv"), index=False)


if __name__ == "__main__":
    main()
