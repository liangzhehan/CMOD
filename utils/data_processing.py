import numpy as np


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, n_nodes):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.n_interactions = len(sources)
        self.unique_nodes = set(list(range(n_nodes)))
        self.n_unique_nodes = n_nodes


def upper_bound(nums, target):
    l, r = 0, len(nums) - 1
    pos = -1
    while l <= r:
        mid = int((l + r) / 2)
        if nums[mid] > target:
            r = mid - 1
            pos = mid
        else:  # >
            l = mid + 1
    return pos


def get_od_data(config):
    whole_data = np.load(config["data_path"]).astype("int").reshape([-1, 3])
    od_matrix = np.load(config["matrix_path"])
    back_points = np.load(config["point_path"])
    print("data loaded")
    all_time = (config["train_day"] + config["val_day"] + config["test_day"]) * config["day_cycle"]
    val_time, test_time = (config["train_day"]) * config["day_cycle"] - config["input_len"], (config["train_day"] + config["val_day"]) * config["day_cycle"] - config["input_len"]
    sources = whole_data[:, 0]
    destinations = whole_data[:, 1]
    timestamps = whole_data[:, 2]
    edge_idxs = np.arange(whole_data.shape[0])
    n_nodes = config["n_nodes"]
    node_features = np.diag(np.ones(n_nodes))
    full_data = Data(sources, destinations, timestamps, edge_idxs, n_nodes)
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))

    return n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix, back_points

