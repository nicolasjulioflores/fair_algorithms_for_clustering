import configparser
import time
from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
from cplex_fair_assignment_lp_solver import fair_partial_assignment
from cplex_violating_clustering_lp_solver import violating_lp_clustering
from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key,
                                 vanilla_clustering, write_fairness_trial)
from util.configutil import read_list


# This function takes a dataset and performs a fair clustering on it.
# Arguments:
#   dataset (str) : dataset to use
#   config_file (str) : config file to use (will be read by ConfigParser)
#   data_dir (str) : path to write output
#   num_clusters (int) : number of clusters to use
#   deltas (list[float]) : delta to use to tune alpha, beta for each color
#   max_points (int ; default = 0) : if the number of points in the dataset 
#       exceeds this number, the dataset will be subsampled to this amount.
# Output:
#   None (Writes to file in `data_dir`)  
def fair_clustering(dataset, config_file, data_dir, num_clusters, deltas, max_points, violating, violation):
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Read data in from a given csv_file found in config
    # df (pd.DataFrame) : holds the data
    df = read_data(config, dataset)

    # Subsample data if needed
    if max_points and len(df) > max_points:
       df = subsample_data(df, max_points)

    # Clean the data (bucketize text data)
    df, _ = clean_data(df, config, dataset)

    # variable_of_interest (list[str]) : variables that we would like to collect statistics for
    variable_of_interest = config[dataset].getlist("variable_of_interest")

    # Assign each data point to a color, based on config file
    # attributes (dict[str -> defaultdict[int -> list[int]]]) : holds indices of points for each color class
    # color_flag (dict[str -> list[int]]) : holds map from point to color class it belongs to (reverse of `attributes`)
    attributes, color_flag = {}, {}
    for variable in variable_of_interest:
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)
        
        condition_str = variable + "_conditions"
        bucket_conditions = config[dataset].getlist(condition_str)

        # For each row, if the row passes the bucket condition, 
        # then the row is added to that color class
        for i, row in df.iterrows():
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    colors[bucket_idx].append(i)
                    this_color_flag[i] = bucket_idx

        attributes[variable] = colors
        color_flag[variable] = this_color_flag

    # representation (dict[str -> dict[int -> float]]) : representation of each color compared to the whole dataset
    representation = {}
    for var, bucket_dict in attributes.items():
        representation[var] = {k : (len(bucket_dict[k]) / len(df)) for k in bucket_dict.keys()}

    # Select only the desired columns
    selected_columns = config[dataset].getlist("columns")
    df = df[[col for col in selected_columns]]

    # Scale data if desired
    scaling = config["DEFAULT"].getboolean("scaling")
    if scaling:
        df = scale_data(df)

    # Cluster the data -- using the objective specified by clustering_method
    clustering_method = config["DEFAULT"]["clustering_method"]

    if not violating:
        t1 = time.monotonic()
        initial_score, pred, cluster_centers = vanilla_clustering(df, num_clusters, clustering_method)
        t2 = time.monotonic()
        cluster_time = t2 - t1
        print("Clustering time: {}".format(cluster_time))
        
        ### Calculate fairness statistics
        # fairness ( dict[str -> defaultdict[int-> defaultdict[int -> int]]] )
        # fairness : is used to hold how much of each color belongs to each cluster
        fairness = {}
        # For each point in the dataset, assign it to the cluster and color it belongs too
        for attr, colors in attributes.items():
            fairness[attr] = defaultdict(partial(defaultdict, int))
            for i, row in enumerate(df.iterrows()):
                cluster = pred[i]
                for color in colors:
                    if i in colors[color]:
                        fairness[attr][cluster][color] += 1
                        continue

        # sizes (list[int]) : sizes of clusters
        sizes = [0 for _ in range(num_clusters)]
        for p in pred:
            sizes[p] += 1

        # ratios (dict[str -> dict[int -> list[float]]]): Ratios for colors in a cluster
        ratios = {}
        for attr, colors in attributes.items():
            attr_ratio = {}
            for cluster in range(num_clusters):
                attr_ratio[cluster] = [fairness[attr][cluster][color] / sizes[cluster] 
                                for color in sorted(colors.keys())]
            ratios[attr] = attr_ratio
    else:
        # These added so that output format is consistent among violating and
        # non-violating trials
        cluster_time, initial_score = 0, 0
        fairness, ratios = {}, {}
        sizes, cluster_centers = [], []

    # dataset_ratio : Ratios for colors in the dataset
    dataset_ratio = {}
    for attr, color_dict in attributes.items():
        dataset_ratio[attr] = {int(color) : len(points_in_color) / len(df) 
                            for color, points_in_color in color_dict.items()}

    # fairness_vars (list[str]) : Variables to perform fairness balancing on
    fairness_vars = config[dataset].getlist("fairness_variable")
    for delta in deltas:
        #   alpha_i = a_val * (representation of color i in dataset)
        #   beta_i  = b_val * (representation of color i in dataset)
        alpha, beta = {}, {}
        a_val, b_val = 1 / (1 - delta), 1 - delta
        for var, bucket_dict in attributes.items():
            alpha[var] = {k : a_val * representation[var][k] for k in bucket_dict.keys()}
            beta[var] = {k : b_val * representation[var][k] for k in bucket_dict.keys()}

        # Only include the entries for the variables we want to perform fairness on
        # (in `fairness_vars`). The others are kept for statistics.
        fp_color_flag, fp_alpha, fp_beta = (take_by_key(color_flag, fairness_vars),
                                            take_by_key(alpha, fairness_vars),
                                            take_by_key(beta, fairness_vars))

        # Solves partial assignment and then performs rounding to get integral assignment
        if not violating:
            t1 = time.monotonic()
            res = fair_partial_assignment(df, cluster_centers, fp_alpha, fp_beta, fp_color_flag, clustering_method)
            t2 = time.monotonic()
            lp_time = t2 - t1

        else:
            t1 = time.monotonic()
            res = violating_lp_clustering(df, num_clusters, fp_alpha, fp_beta, fp_color_flag, clustering_method, violation)
            t2 = time.monotonic()
            lp_time = t2 - t1

            # Added so that output formatting is consistent among violating
            # and non-violating trials
            res["partial_objective"] = 0
            res["partial_assignment"] = []

        ### Output / Writing data to a file
        # output is a dictionary which will hold the data to be written to the
        #   outfile as key-value pairs. Outfile will be written in JSON format.
        output = {}

        # num_clusters for re-running trial
        output["num_clusters"] = num_clusters

        # Whether or not the LP found a solution
        output["success"] = res["success"]

        # Nonzero status -> error occurred
        output["status"] = res["status"]
        
        output["dataset_distribution"] = dataset_ratio

        # Save alphas and betas from trials
        output["alpha"] = alpha
        output["beta"] = beta

        # Save original clustering score
        output["unfair_score"] = initial_score

        # Clustering score after addition of fairness
        output["fair_score"] = res["objective"]
        
        # Clustering score after initial LP
        output["partial_fair_score"] = res["partial_objective"]

        # Save size of each cluster
        output["sizes"] = sizes

        output["attributes"] = attributes

        # Save the ratio of each color in its cluster
        output["ratios"] = ratios

        # These included at end because their data is large
        # Save points, colors for re-running trial
        # Partial assignments -- list bc. ndarray not serializable
        output["centers"] = [list(center) for center in cluster_centers]
        output["points"] = [list(point) for point in df.values]
        output["assignment"] = res["assignment"]

        output["partial_assignment"] = res["partial_assignment"]

        output["name"] = dataset
        output["clustering_method"] = clustering_method
        output["scaling"] = scaling
        output["delta"] = delta
        output["time"] = lp_time
        output["cluster_time"] = cluster_time
        output["violating"] = violating
        output["violation"] = violation

        # Writes the data in `output` to a file in data_dir
        write_fairness_trial(output, data_dir)

        # Added because sometimes the LP for the next iteration solves so 
        # fast that `write_fairness_trial` cannot write to disk
        time.sleep(1) 
