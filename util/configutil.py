'''
Functions that help with reading from a config file.
'''
# Reads the given config string in as a list
#   Allows for multi-line lists.
def read_list(config_string, delimiter=','):
    config_list = config_string.replace("\n", "").split(delimiter)
    return [s.strip() for s in config_list]

# Read a given range from config string in as a list
def read_range(config_string, delimiter=','):
    start, end, step = tuple(map(int, config_string.split(delimiter)))
    return list(range(start, end, step))

# Make sure that the given config uses one of the available config methods.
def validate_method(config):
    available_methods = {"kmeans", "kmedians", "kcenters"}
    if config["DEFAULT"]["clustering_method"] not in available_methods:
        print("Not a valid clustering method. Available methods are: " 
                "\'kmeans\' and \'kmedians\'.")
        return False
    else:
        return True