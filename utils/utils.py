import numpy as np
import torch

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=False, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round, (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance


def to_device(data, device):
    if isinstance(data, list):
        return [to_device(d, device) for d in data]
    elif isinstance(data, np.ndarray):
        return torch.Tensor(data).to(device)
    elif isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data.keys()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.Tensor(data).to(device)
