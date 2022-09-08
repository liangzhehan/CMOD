import torch
from torch import nn


class Memory_lambs(nn.Module):

    def __init__(self, n_nodes, memory_dimension, lambs, device="cpu"):
        super(Memory_lambs, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.lambs = lambs
        self.lamb_len = lambs.shape[0]
        self.device = device
        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.lamb_len, self.memory_dimension)),
                                   requires_grad=False).to(self.device)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes), requires_grad=False).to(self.device)

    def get_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    def detach_memory(self):
        self.memory.detach_()


class ExpMemory_lambs(nn.Module):
    def __init__(self, n_nodes, memory_dimension, lambs, device="cpu"):
        super(ExpMemory_lambs, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.lambs = lambs
        self.lamb_len = lambs.shape[0]
        self.device = device
        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        self.memory = nn.Parameter(torch.cat([torch.zeros((self.n_nodes, self.lamb_len, self.memory_dimension)),
                                              torch.ones((self.n_nodes, self.lamb_len, 1))], dim=2),
                                   requires_grad=False).to(self.device)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes), requires_grad=False).to(self.device)

    def get_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    def detach_memory(self):
        self.memory.detach_()
