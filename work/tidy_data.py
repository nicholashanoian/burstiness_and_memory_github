"""
Nicholas Hanoian
3/29/2020

GitHub Team Dynamics -- read cached data from github archive and store
in 3 tidy data csvs:

- pushes.csv: a record of all push events in the data. Columns:
  timestamp, user, repo_id, size, location

- repository_stats.csv: statistics about all repositories observed in
the data. Columns: repo_id, stargazers, max_stargazers, size, watchers, forks,
url

- team_members.csv: record of all members of all teams. Columns:
  repo_id, user

"""

import os
import gzip
import json
import csv


from collections import defaultdict
from datetime import date, timedelta
import time
import argparse

from tqdm import tqdm


def get_args():
    """get command line arguments for input_dir, output_dir, start_date, and end_date"""

    # default folder containing data
    # default_input_dir = "/home/nick/downloads/gharchive"
    default_input_dir = "/Volumes/BL_DATASTORE/DATASETS/github-archive"

    default_output_dir = "."

    # default dates

    # original data window
    default_start_date = "2013-01-01"
    default_end_date = "2014-04-01"

    # march subset for testing
    # default_start_date = "2014-03-01"
    # default_end_date = "2014-03-02"

    parser = argparse.ArgumentParser(
        description="Convert data from gharchive.org to three tidy data files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input and output directory args
    parser.add_argument("-i", "--input-dir", help="Input directory",
                        default=default_input_dir, dest="input_dir")
    parser.add_argument("-o", "--output-dir", help="Output directory",
                        default=default_output_dir, dest="output_dir")

    # date args
    parser.add_argument('-s', "--start-date", help="Start date - format YYYY-MM-DD",
                        type=date.fromisoformat, default=default_start_date, dest="start_date")
    parser.add_argument('-e', "--end-date", help="End date format - YYYY-MM-DD (Inclusive)",
                        type=date.fromisoformat, default=default_end_date, dest="end_date")

    return parser.parse_args()


# ====================================================================
# Functions to save data to csvs
# ====================================================================

def write_csv(data, keys, filename):
    """write a csv to filename given list of dictionaries and list of keys
    to include from those dictionaries, should they exist"""
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in data:
            writer.writerow([row[k] if k in row else "" for k in keys])


def save_repositories(repositories, filename):
    """save repository data to csv"""
    keys = ["repo_id", "stargazers", "max_stargazers", "size", "watchers", "forks", "created_at", "url"]
    repos_with_id = []
    for repo_id, stats in repositories.items():
        stats["repo_id"] = repo_id
        repos_with_id.append(stats)
    write_csv(repos_with_id, keys, filename)


def save_pushes(pushes, filename):
    """save data for each push to a csv"""
    keys = ["timestamp", "user", "repo_id", "size", "location"]
    write_csv(pushes, keys, filename)


def save_repos_to_teams(repos_to_teams, filename):
    """save a csv with repo_id, user for each member of each team"""
    keys = ["repo_id", "user"]
    repos_members = []
    for repo_id, members in repos_to_teams.items():
        for member in members:
            repos_members.append({"repo_id": repo_id, "user": member})
    write_csv(repos_members, keys, filename)


def write_event_counts(counts, file_handle):
    """to write dictionaries of event counts to log file"""
    for event_type, count in counts.items():
        file_handle.write(f"{event_type}: {count}\n")


# ====================================================================
# Functions to read in and parse files
# ====================================================================

def get_filenames(path, start_date, end_date):
    """get list of filenames to read in, both ends are inclusive"""
    filenames = []
    # Starting in April of 2014, the format changes from 2014-03-31-01 to 2014-04-01-1
    # i.e. hour is not zero padded starting in April 2014
    format_switch_date = date(2014, 4, 1)

    while start_date <= end_date:
        # iterate through all hours
        for hour in range(24):
            # add filename depending on if before change date or not
            if start_date < format_switch_date:
                filenames.append(
                    os.path.join(path, f"{start_date.strftime('%Y-%m-%d')}-{hour:02}.json.gz"))
            else:
                filenames.append(
                    os.path.join(path, f"{start_date.strftime('%Y-%m-%d')}-{hour}.json.gz"))
        # move to next day
        start_date += timedelta(days=1)

    return filenames


def update_repository(repo, repositories):
    """update the repositories dictionary with new info for a repo found
    while reading in a file"""
    wanted_keys = ["stargazers", "size", "watchers", "forks", "created_at", "url"]

    repo_id = repo["id"]
    # get just the key, value pairs we want from repo
    repo_subset = dict((k, repo[k]) for k in wanted_keys if k in repo)

    if repo_id in repositories:
        max_stargazers = max(repositories[repo_id]["max_stargazers"], repo_subset["stargazers"])
        repo_subset["max_stargazers"] = max_stargazers
        repositories[repo_id].update(repo_subset)
    else:
        repo_subset["max_stargazers"] = repo_subset["stargazers"]
        repositories[repo_id] = repo_subset
    return repositories


def handle_file(filename, data):
    """Read in one file for one hour of github activity data, and update
    data accordingly."""

    with gzip.open(filename, "rb") as f:
        for line in f:
            try:
                # parse json

                # so we can gracefully handle non-unicode characters
                decoded_line = line.decode(errors="replace")
                event = json.loads(decoded_line)

                # if the event has a repo and an event type
                if ("repository" in event) and ("type" in event):

                    # track counts of events with repo attribute
                    data["events_with_repo"][event["type"]] += 1

                    # update repository stats
                    repo = event["repository"]
                    repo_id = repo["id"]
                    data["repositories"] = update_repository(repo, data["repositories"])

                    # if it's a push, add to list of pushes, and add the
                    # pusher to that repo's team
                    if event["type"] == "PushEvent":

                        # try to get location, otherwise blank
                        if ("actor_attributes" in event) and \
                           ("location" in event["actor_attributes"]):
                            location = event["actor_attributes"]["location"]
                        else:
                            location = ""

                        # add to list of pushes
                        data["pushes"].append({"timestamp": event["created_at"],
                                               "user":      event["actor"],
                                               "repo_id":   repo_id,
                                               "size":      event["payload"]["size"],
                                               "location":  location})

                        data["repos_to_teams"][repo_id].add(event["actor"])

                    # for push events, create (creating branches and
                    # tags) events, and public events, add that user
                    # to the team
                    # if event["type"] in ["PushEvent", "CreateEvent", "PublicEvent"]:
                        

                    # event type for when a user accepts an invitation to a team
                    # ---> Excluded for now, not sure if they should be included
                    # if event["type"] == "MemberEvent":
                        # if event["payload"]["action"] == "added": # member is added
                        # data["repos_to_teams"][repo_id].add(event["payload"]["member"]["login"])
                        # elif event["payload"]["action"] == "removed": # member is removed
                        # data["repos_to_teams"][repo_id].remove(event["payload"]["member"]["login"])

                elif "type" in event:
                    # track what types of events don't have repo attached
                    data["events_without_repo"][event["type"]] += 1


            except Exception as err:
                print(f"Error parsing {filename}: {err}")
                data["errors"].append(f"Error parsing {filename}: {err}")

    return data


def log_info_to_file(data, filename):
    """Log info about the number of events of each type for events with
    and without repo data, and filenames that were not found"""

    with open(filename, "w") as logfile:
        logfile.write(f"Dates analyzed: {data['start_date'].isoformat()}"
                      f" -- {data['end_date'].isoformat()}\n")
        logfile.write(f"Run time: {time.time() - data['start_time']:.2f}s\n")

        # counts of events with repos
        logfile.write("\nevents_with_repo:\n")
        write_event_counts(data["events_with_repo"], logfile)

        # counts of events without repo
        logfile.write("\nevents_without_repo:\n")
        write_event_counts(data["events_without_repo"], logfile)

        # files we did not find
        logfile.write("\nmissing_files:\n")
        for missing_filename in data["missing_files"]:
            logfile.write(missing_filename + "\n")

        # parsing errors
        logfile.write("\nparsing errors:\n")
        for err in data["errors"]:
            logfile.write(err + "\n")


# ====================================================================
# Main
# ====================================================================

def main():

    args = get_args()

    # initialize datastructures
    data = {}

    # dictionary of number of occurrences of each event type which has a repo attribute
    data["events_with_repo"] = defaultdict(int)
    # dictionary of number of occurrences of each event type which do not have a repo attribute
    data["events_without_repo"] = defaultdict(int)

    # dictionary mapping repo_id to a dictionary of stats about the
    # repo. See update_repository function above
    data["repositories"] = {}
    # list of dictionaries of info about pushes
    data["pushes"] = []
    # dictionary mapping repo_ids to sets of team members
    data["repos_to_teams"] = defaultdict(set)

    # track files we did not find
    data["missing_files"] = []
    data["errors"] = []

    # extra info to log
    data["start_time"] = time.time()
    data["start_date"] = args.start_date
    data["end_date"] = args.end_date

    # list of filenames to read
    filenames = get_filenames(args.input_dir, args.start_date, args.end_date)

    # read in and parse data
    for filename in tqdm(filenames):
        # only try to read if we know the file exists
        if os.path.exists(filename):
            data = handle_file(filename, data)
        else:
            data["missing_files"].append(filename)
            print(f"Error: {filename} not found. Continuing.")

    # make output directory if necessary
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # write data to csvs
    save_repositories(data["repositories"],
                      os.path.join(args.output_dir, "repository_stats.csv"))
    save_pushes(data["pushes"],
                os.path.join(args.output_dir, "pushes.csv"))
    save_repos_to_teams(data["repos_to_teams"],
                        os.path.join(args.output_dir, "team_members.csv"))

    # log extra info
    log_info_to_file(data, os.path.join(args.output_dir, "tidy_data_log.txt"))


if __name__ == "__main__":
    main()
