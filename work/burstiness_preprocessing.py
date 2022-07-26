import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import gamma
from tqdm import tqdm

from scipy.stats import weibull_min, lognorm, exponweib
import powerlaw


from preprocessing import initial_filter





def taus_dict_to_txt(taus_dict, filename):
    """write dict of the form id:[tau0, tau1, tau2, ...] to file"""
    with open(filename, "w") as f:
        for key, taus in tqdm(taus_dict.items()):
            if taus != []:
                f.write(str(key) + ":")
                for tau in taus:
                    f.write(str(tau) + " ")
                f.write("\n")


def taus_by_repo(pushes):
    # caring about just repos
    repos = pushes.groupby("repo_id")
    repos_to_taus = {}
    for repo_id, group in tqdm(repos):
        times = group.sort_values("timestamp")["timestamp"]
        taus = times[1:].reset_index(drop=True) - times[:-1].reset_index(drop=True)
        taus = [t.total_seconds() / (60*60) for t in taus]
        repos_to_taus[repo_id] = taus

    return repos_to_taus


def taus_by_user(pushes):
    users = pushes.groupby("user")
    users_to_taus = {}
    for user, group in tqdm(users):
        if len(group) >= 30:
            times = group.sort_values("timestamp")["timestamp"]
            taus = times[1:].reset_index(drop=True) - times[:-1].reset_index(drop=True)
            taus = [t.total_seconds() / (60*60) for t in taus]
            users_to_taus[user] = taus
    return users_to_taus


def calc_top_users_stats(users, pushes, repository_stats):
    user_to_avg_success = {}
    top_user_pushes = pushes[pushes["user"].isin(users)]

    for user, group in tqdm(top_user_pushes.groupby("user")):
        projects = group["repo_id"].unique()
        stargazers = repository_stats[repository_stats["repo_id"].isin(projects)]["max_stargazers"]
        user_to_avg_success[user] = {"max": stargazers.max(), "avg": stargazers.mean(),
                                     "n_repo": len(stargazers)}

    return user_to_avg_success


# def main():


directory = "/home/nick/apps/data/github-burstiness/04-22/"
print("Reading in data from", directory)
all_repository_stats = pd.read_csv(os.path.join(directory, "repository_stats.csv"))
all_pushes = pd.read_csv(os.path.join(directory, "pushes.csv"))
all_team_members = pd.read_csv(os.path.join(directory, "team_members.csv"))

print("Filtering out insignificant teams")
_, pushes, _ = initial_filter(all_repository_stats, all_team_members, all_pushes)


repository_stats = pd.read_csv(os.path.join(directory, "repository_stats_expanded.csv"))

# parse dates
timestamps = pd.to_datetime(pushes["timestamp"], utc=True)
timestamps = pd.DatetimeIndex(timestamps).tz_convert(None)

pushes["timestamp"] = timestamps

repos_to_taus = taus_by_repo(pushes)
taus_dict_to_txt(repos_to_taus, os.path.join(directory, "repos_to_taus.txt"))

users_to_taus = taus_by_user(pushes)
taus_dict_to_txt(users_to_taus, os.path.join(directory, "users_to_taus.txt"))

top_users_stats = calc_top_users_stats(users_to_taus.keys(), pushes, repository_stats)
top_users_stats_df = pd.DataFrame([{"user": key, "max_stargazers": value["max"],
                                        "avg": value["avg"], "n_repo": value["n_repo"]}
                                       for key, value in top_users_stats.items()])
top_users_stats_df.to_csv(os.path.join(directory, "top_users_stats.csv"), index=False)


def analyze_all_taus(repos_to_taus):
    all_taus = []
    for repo in repos_to_taus:
        all_taus += repos_to_taus[repo]
    all_taus = pd.Series(all_taus)

    fig, ax = plt.subplots()
    ax.hist(all_taus, bins=200, log=True)
    fig.show()




    x = [1,1,2,3,4,2,1,1, 20, 200]

    results = powerlaw.Fit(all_taus.sample(frac=0.01))
    fig1 = results.plot_ccdf(linewidth=3)
    results.stretched_exponential.plot_ccdf(ax=fig1, color="g", linestyle="--")
    results.lognormal.plot_ccdf(ax=fig1, color="r", linestyle="--")
    results.power_law.plot_ccdf(ax=fig1, color="orange", linestyle="--")
    results.truncated_power_law.plot_ccdf(ax=fig1, color="purple", linestyle="--")
    # results.lognormal_positive.plot_ccdf(ax=fig1, color="black", linestyle="--")

    results.distribution_compare("lognormal", "power_law")
    results.distribution_compare("lognormal", "truncated_power_law")
    results.distribution_compare("stretched_exponential", "truncated_power_law")
    results.distribution_compare("stretched_exponential", "lognormal")
    plt.legend()
    # fig1.legend()
    plt.show()



n_pushes = pushes["user"].value_counts()

fig, ax = plt.subplots()
ax.hist(n_pushes, bins=200, log=True)
fig.show()



n_repos = []
for user, group in pushes.groupby("user"):
    n_repos.append(len(group) >= 30)


fig, ax = plt.subplots()
ax.hist(n_repos, bins=200, log=True)
fig.show()

