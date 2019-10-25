import datetime
import json
import random
from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from pyclustering.cluster.kmedians import kmedians
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Read data in from a given csv_file found in config
# Arguments:
#   config (ConfigParser) : config specification (dict-like)
#   dataset (str) : name of dataset in config file to read from
# Output:
#   df (pd.DataFrame) : contains data from csv_file in `config`
def read_data(config, dataset):
    csv_file = config[dataset]["csv_file"]
    df = pd.read_csv(csv_file, sep=config[dataset]["separator"])

    if config["DEFAULT"].getboolean("describe"):
        print(df.describe())

    return df

# Clean the data. Bucketize text data, convert int to float.
# Arguments:
#   df (pd.DataFrame) : DataFrame containing the data
#   config (ConfigParser) : Config object containing the specifications of
#       which columns are text.
#   dataset (str) : Name of dataset being used.
def clean_data(df, config, dataset):
    # CLEAN data -- only keep columns as specified by the config file
    selected_columns = config[dataset].getlist("columns")
    variables_of_interest = config[dataset].getlist("variable_of_interest")

    # Bucketize text data
    text_columns = config[dataset].getlist("text_columns", [])
    for col in text_columns:
        # Cat codes is the 'category code'. Aka it creates integer buckets automatically.
        df[col] = df[col].astype('category').cat.codes

    # Remove the unnecessary columns. Save the variable of interest column, in case
    # it is not used for clustering.
    variable_columns = [df[var] for var in variables_of_interest]
    # df = df[[col for col in selected_columns]]

    # Convert to float, otherwise JSON cannot serialize int64
    for col in df:
        if col in text_columns or col not in selected_columns: continue
        df[col] = df[col].astype(float)

    if config["DEFAULT"].getboolean("describe_selected"):
        print(df.describe())

    return df, variable_columns

# Return a df with N subsamples from df
# Arguments:
#   df (pd.DataFrame) : Dataframe to subsample
#   N (int) : number of samples to take
# Output:
#   df (pd.DataFrame) : Subsampled Dataframe
def subsample_data(df, N):
    return df.sample(n=N).reset_index(drop=True)

# Scales each of the columns to have mean = 0, var = 1
# Arguments:
#   df (pd.DataFrame) : Dataframe to scale
# Output:
#   df (pd.DataFrame) : Dataframe scaled
def scale_data(df):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
    return df

# Perform the K-means clustering on the dataframe. Return the centers of the k-means clustering.
# Add back the variable_of_interest column to the dataframe.
# Arguments:
#   df (pd.DataFrame) : Dataframe to perform clustering on
#   variable_column (pd.Series) : Column that fairness is to be performed on. Clustering
#       should not be done on this column.
#   config (ConfigParser) : Config file that is being used for the run.
#   dataset (str) : Name of the dataset, used to key the config.
def get_cluster_centers(df, variable_column, config, dataset):
    n_clusters = config["DEFAULT"].getint("n_clusters")
    variable_of_interest = config[dataset]["variable_of_interest"]

    # Fit kmeans to the data
    data = df.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)

    # Add variable of interest column back in case it was removed
    df[variable_of_interest] = variable_column

    return kmeans.cluster_centers_

# Return a dict with the key, value pairs from a dictionary (`dic`) where the keys 
# are in a sequence provided (`seq`).
# Arguments :
#   dic (dict) : Dictionary to take key, value pairs from
#   seq (iterable) : contains keys to keep from `dic`
def take_by_key(dic, seq):
    return {k : v for k, v in dic.items() if k in seq}

# Dump output to a file in JSON format
# Arguments : 
#   output (dict) : Data to write
#   data_dir (str) : Where to write the data
def write_fairness_trial(output, data_dir, post_fix = ''):
    g_date_format = "%Y-%m-%d-%H:%M:%S" # format of output
    now = datetime.datetime.now().strftime(g_date_format)
    data_file = data_dir + post_fix + now
    with open(data_file, "w") as dataf:
        json.dump(output, dataf)

# Returns centers in a sorted format so that they will be more consistent
#   across trials for comparing relative sizes of clusterings. Although
#   exact Euclidean centers are not the same usually, they are roughly similar.
# Arguments:
#   centers (nd.array) : centers, returned from vanilla clustering, that will be sorted
def sort_centers(centers):
    tup_centers = sorted([tuple(center) for center in centers])
    # return to list in case anything depended on it
    return [list(center) for center in tup_centers]

# # Plot the change of some costs with respect to number of clusters
# # Arguments:
# #     num_of_clusters: A list of integers, including multiple number of clusters
# #     costs: A dictionary of lists. The keys are used in the plot legend and each list is of matching length with num_of_clusters
# #     markers: A list of marker characters used for plotting. Length must match the number of keys in costs
# #     plot_name: A string used for plot title and also for naming the output png file
# #     output_dir: Self explanatory :)
# def plot_cost_stats(num_of_clusters, costs, markers, plot_title, config, destination):

#     plt.clf()
#     plt.autoscale(enable=True, axis='both', tight=True)

#     colors = cm.get_cmap("Set1").colors
#     labels = [name for name in costs]
#     key = []
#     for name, color, marker in zip(labels, colors, markers):
#         plt.plot(num_of_clusters, costs[name], linestyle='-', marker=marker, color=color)
#         key.append(mpatches.Patch(color=color, label=name))

#     plt.title(plot_title)
#     plt.xlabel('Number of Clusters')
#     plt.ylabel('Clustering Cost')
#     plt.legend(handles=key, loc='upper right')

#     if config["DEFAULT"].getboolean("save_plot"):
#         plt.savefig(destination)
#     else:
#         plt.show()


# This is a 2-approximation greedy algorithm for the k-center problem by Gonzales.
# Arguments:
#     df: dataframe of the points
#     n_clusters : the ``k'' in the k-center problem
# Outputs:
#     max_dist: The cost of the solution
#     pred: The index of the cluster each points belongs to, in range [0,k-1]
#     cluster_centers: The k points picked as cluster centers
def gonzales_k_center(df, n_clusters):

    def squared_euclidean_distance(point1, point2):
        return sum([(f1 - f2) ** 2 for f1, f2 in zip(point1, point2)])

    nr_points = len(df.index)

    # The indices of the points picked as centers
    cluster_centers = []
    # For each point, holds the index of the cluster it is assigned to
    pred = [None] * nr_points
    # For each point, holds its distance to the cluster it is assigned to
    distance_to_sln = [None] * nr_points
    # Index of a point with maximum distance to this solution, set to an arbitrary index for the first run
    max_dist_ind = 0
    # Maximum distance of a point to a cluster center
    max_dist = 0

    # Add more centers to the solution
    for i in range(n_clusters):
        # Add the farthest point to the solution, to the solution!
        cluster_centers.append(df.iloc[max_dist_ind].data.tolist())

        # Reset max_dist and max_dist_ind
        max_dist = 0
        max_dist_ind = 0
        for row in df.iterrows():
            i_point, data = row
            point = data.tolist()

            # Distance to the the newly added center
            new_point_dist = squared_euclidean_distance(point, cluster_centers[i])

            # If the current point is closer to the newly added center compared to the rest of the centers
            if i == 0 or new_point_dist < distance_to_sln[i_point]:
                # Assign it to this new center
                pred[i_point] = i
                # Update its connection cost
                distance_to_sln[i_point] = new_point_dist


            # Maintain a running max distance point
            if distance_to_sln[i_point] > max_dist:
                max_dist = distance_to_sln[i_point]
                max_dist_ind = i_point


    return np.sqrt(max_dist), pred, cluster_centers


# This is a 5-approximation local-search algorithm for the k-median problem by Arya et.al.
# The initial centers are picked with the distance-weighting method similar to k-means++
# Arguments:
#     df: dataframe of the points
#     n_clusters : the ``k'' in the k-center problem
#     num_trial: The algorithm is run for num_trial many times and returns the best solution found
# Outputs:
#     best_cost: The cost of the solution
#     best_pred: The index of the cluster each points belongs to, in range [0,k-1]
#     best_cluster_centers: The k points picked as cluster centers
def arya_etal_k_median(df, n_clusters, num_trial = 1):

    if n_clusters < 2:
        raise Exception("Current implementation of k-median does not support n_clusters = {} < 2".format(n_clusters))

    all_pair_distance = squareform(pdist(df.values,'euclidean'))
    nr_points = len(df.index)

    best_cluster_centers = [None] * n_clusters
    best_pred = [None] * nr_points
    best_cost = None

    # As picking the initial centers involves some random choices,
    # this loop is trying to boost the possibility of landing on a good initial set of centers,
    # by repeating the whole process for num_trial many times
    for trial in range(0 , num_trial):
        # The indices of the points picked as centers
        cluster_centers = []
        # Accumulative probabilities of picking points as centers
        accumulative_prob = np.cumsum([1/nr_points] * nr_points)
        # Distance of points to (intermediate) solution
        weights = [None] * nr_points

        for c in range(0, n_clusters):
            new_c = None
            # Pick a new point as center (no duplicates)
            while new_c is None or new_c in cluster_centers:
                # Pick a center with probability relative to accumulative_prob
                # The 1e-9 is there to account for floating point computation errors,
                # because we need to make sure rand <= the last element in accumulative_prob
                rand = random.uniform(0, 1) - 1e-9
                new_c = np.searchsorted(accumulative_prob, rand)
            cluster_centers.append(new_c)

            running_sum = 0
            accumulative_prob = []
            for p in range(0, nr_points):
                # Update the distance to current solution
                if c == 0 or all_pair_distance[p][cluster_centers[c]] < weights[p]:
                    weights[p] = all_pair_distance[p][cluster_centers[c]]

                # Update accumulative_prob according to new weights
                running_sum = running_sum + weights[p]
                accumulative_prob.append(running_sum)
            accumulative_prob = np.divide(accumulative_prob, running_sum)


        # For each point, holds the index of the cluster it is assigned to
        pred = [None] * nr_points
        pred_susbstitute = [None] * nr_points
        cost = 0

        # Loop over the improvement threshold
        # A swap operation is applied only if its cost is < (1-1/(2**iter)) times the current cost
        # Over each iteration, we allow for smaller changes
        for iter in range (2,5):

            updated_sln = True
            while updated_sln is True:
                updated_sln = False
                # Assign each point to the closest center and compute the solution cost
                cost = 0
                for p in range(0,nr_points):
                    # center_distances = np.array([all_pair_distance[p][cluster_centers[c]] for c in range(n_clusters)])
                    # pred[p] = np.argmin(center_distances)
                    # connection_cost = center_distances[pred[p]]
                    #
                    # np.delete(center_distances, pred[p])
                    # pred_susbstitute[p] = np.argmin(center_distances)
                    # if pred_susbstitute[p] >= pred[p]:
                    #     pred_susbstitute[p] = pred_susbstitute[p] + 1
                    # print("pred {} sub {}".format(pred[p] , pred_susbstitute[p]))

                    pred[p] = 0
                    pred_susbstitute[p] = None
                    connection_cost = all_pair_distance[p][cluster_centers[0]]

                    for c in range(1, n_clusters):
                        if all_pair_distance[p][cluster_centers[c]] < connection_cost:
                            # The previously closest center is now the second-closest center
                            pred_susbstitute[p] = pred[p]
                            # Update the closest center
                            pred[p] = c
                            connection_cost = all_pair_distance[p][cluster_centers[c]]
                        if pred_susbstitute[p] is None:
                            pred_susbstitute[p] = c
                    cost = cost + connection_cost

                # For all possible new centers new_c
                for new_c in range(0,nr_points):
                    # Running cost of swapping new_c with each of the current centers
                    swap_cost = np.array([0] * n_clusters)

                    # For all points, compute the connection cost of swapping new_c with each one of the current centers
                    for p in range(0,nr_points):
                        connection_cost = np.array([all_pair_distance[p][new_c]]* n_clusters)
                        # If p does not go to this new_c, it has to go to pred[p]
                        c = cluster_centers[pred[p]]
                        if all_pair_distance[p][new_c] > all_pair_distance[p][c]:
                            connection_cost = np.array([all_pair_distance[p][c]]* n_clusters)
                            sub_c = cluster_centers[pred_susbstitute[p]]
                            # But if pred[p] is thrown away, p has to choose between sub_c and new_c
                            connection_cost[pred[p]] = min(all_pair_distance[p][new_c], all_pair_distance[p][sub_c])
                        swap_cost = np.add(swap_cost, connection_cost)

                    # Find the center for which the swapping cost of new_c is minimum
                    new_cost, c = min((new_cost, c) for (c, new_cost) in enumerate(swap_cost))
                    # Check if this new_c is good for substitution
                    if new_cost < (1-1/(2**iter))* cost:
                        cluster_centers[c] = new_c
                        updated_sln = True
                        # Break the loop to allow for iter to be incremented and allow for smaller improvements
                        break

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_pred = pred[:]
            best_cluster_centers = cluster_centers[:]

    actual_centers = []
    for c in best_cluster_centers:
        actual_centers.append(df.iloc[c].data.tolist())

    return best_cost, best_pred, actual_centers


# Calls the appropriate clustering function and returns the solution.
# Arguments:
#     df: dataframe of the points
#     num_clusters: the desired number of clusters
#     clustering_method: Currently implemented clustering methods are "kmeans", "kmedian", and "kcenter"
# Outputs:
#     initial_score: The valueç of the objective function
#     pred: A vector that for each point, has the index of the cluster it is assigned to (starting from 0)
#     cluster_centers: The coordinates of the centers
# Returns None if the given clustering_method is invalid
def vanilla_clustering(df, num_clusters, clustering_method):
    if clustering_method == "kmeans":
        kmeans = KMeans(num_clusters)
        kmeans.fit(df)
        initial_score = np.sqrt(-kmeans.score(df))
        pred = kmeans.predict(df)
        cluster_centers = kmeans.cluster_centers_
        return initial_score, pred, sort_centers(cluster_centers)
    elif clustering_method == "kmedian":
        return arya_etal_k_median(df, num_clusters, 5)
    elif clustering_method == "kcenter":
        return gonzales_k_center(df, num_clusters)

    else:
        raise Exception("Not a valid clustering method. Available methods are: " 
              "\'kmeans\', \'kmedian\', and \'kcenter\'.")
