# README

## Requirements

`Python3.6` is expected for this program because currently the CPLEX solver being used expects that version of Python.

To install the non-CPLEX dependencies, use `pip install -r requirements.txt`.

To install CPLEX, visit the IBM website and navigate to the proper license (student, academic, professional, etc.), and follow the installation guide provided by IBM.

## Running Example Script

Once dependencies are enabled you should be able to run our example code using `python example.py`. The current example script runs on the `bank` dataset with k = 4, using k-means.

## Running your Own Tests

To run one of your own tests, edit the following three things:

1. Create an entry in `example_config.ini`. An entry begins with `[your_title]` and contains all the fields specified by the example.

2. Change the objective in `dataset_configs.ini` if you desire.

3. Run the example script using your specifications instead of example by running `python example.py your_title`.

## Description of Output Format

The output from a trial will be a new file for each run with the timestamp: `%Y-%m-%d-%H:%M:%S`. A run is defined as a combination of `num_cluster` and `delta` in the config file. For example, if two values for `num_clusters` and two deltas are specified, then 4 runs will occur.

Each output file is in JSON format, and can be loaded using the `json` package from the standard library. The data is held as a dictionary format and can be accessed by using string key names of the following fields: 
* `violating` : Whether or not to use the violating LP or the regular fair clustering. True means use the violating LP. False means use fair clustering. The violating LP can typically handle far fewer points because there are more degrees of freedom in this violating LP (because we can choose centers as well). Thus, we recommend that datasets be subsampled to at most 1000, for the violating LP.
* `violating` : Integer that is the maximum number of violations to allow the violating LP to make. From our experimental results, we find that there are no more than 3-4, but theoretically guarantee no more than 4*Delta + 3. Note that this is not the "delta" referenced below, this "capital D" Delta is the amount of color groups that any point can belong to. Typically, this can be calculated as the number of attributes. Reference the paper to learn more.
* `num_clusters` : The number of clusters used for this trial.
* `success` : Whether or not the LP successfully solved, provided by the CPLEX API.
* `status` : Integer status provided by the CPLEX API for the solving of the LP. Non-zero indicates unsuccessful.
* `dataset_distribution`: Dictionary holding distribution of colors in the dataset. First key is the attribute (ie. sex), and second key is the color within that attribute (ie. male).
* `alpha` : Dictionary holding the alphas for various colors. First key is the attribute (ie. sex), and second key is the color within that attribute (ie. male).
* `beta` : Dictionary holding the betas for various colors.
First key is the attribute (ie. sex), and second key is the color within that attribute (ie. male).
* `unfair_score` : Clustering objective score returned by vanilla clustering. 0 if `violating` is True.
* `fair_score` : Clustering objective returned by either the fair clustering or the violating LP.
* `partial_fair_score` : Clustering objective returned by the partial fair clustering LP. 0 if `violating` is True.
* `sizes` : List holding the sizes of the clusters returned by vanilla clustering. Empty list if `violating` is True.
* `attributes` : Dictionary holding the points that belong to each color group. First key is the attribute that is being considered (ie. sex), second key is the color group within that attribute that the point belongs to (ie. male).
* `ratios` : Dictionary holding the ratios of the colors in every cluster. First key is the attribute, second key is cluster, third key is color within attribute.
* `centers` : List of centers found by vanilla clustering. Empty list if `violating` is True.
* `points` : List of points used for fair clustering or violating LP. Useful if the dataset has been subsampled to know which points were chosen by the subsampling method.
* `assignment`: List (sparse) of points and their assigned cluster. There are (# of points) * (# of centers) entries in assignments. For each point `i`, we say that it is assigned to that cluster `f` if `assignment[i*(# of centers) + f] == 1`.
* `partial_assignment`: List (sparse) of partial assignments between points and centers. Has same format as `assignment` however, entries can now be non-integral (ie. between 0 and 1). Empty list if `violating` is True.
* `name` : String name of the dataset chosen. Will use name from `dataset_configs.ini` file.
* `clustering_method` : String name of the clustering method used.
* `scaling` : Boolean of whether or not data was scaled.
* `delta` : delta value used for this run. Note that this is not the overlap but rather the variable involved in the reparameterization of alpha and beta. beta = (1 - delta) and alpha = 1 / (1 - delta).
* `time` : Float that is the time taken for the LP to be solved.
* `cluster_time` : Float that is the time taken for the vanilla clustering to occur. 0 if `violating` is true.
