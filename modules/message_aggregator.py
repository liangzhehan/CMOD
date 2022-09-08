from collections import defaultdict
import torch
import numpy as np


class MessageAggregator(torch.nn.Module):
    """
    Abstract class for the message aggregator module, which given a batch of node ids and
    corresponding messages, aggregates messages with the same node id.
    """

    def __init__(self, device):
        super(MessageAggregator, self).__init__()
        self.device = device

    def aggregate(self, node_ids, messages):
        """
        Given a list of node ids, and a list of messages of the same length, aggregate different
        messages for the same id using one of the possible strategies.
        :param node_ids: A list of node ids of length batch_size
        :param messages: A tensor of shape [batch_size, message_length]
        :param timestamps A tensor of shape [batch_size]
        :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
        """

    def group_by_id(self, node_ids, messages, timestamps):
        node_id_to_messages = defaultdict(list)

        for i, node_id in enumerate(node_ids):
            node_id_to_messages[node_id].append((messages[i], timestamps[i]))

        return node_id_to_messages


class MeanMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(MeanMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages):
        """Only keep the last message for each node"""
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        for node_id in unique_node_ids:
            unique_messages.append(torch.mean(messages[node_id][0], dim=0))
            unique_timestamps.append(messages[node_id][1][-1])

        unique_messages = torch.stack(unique_messages) if len(unique_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(unique_node_ids) > 0 else []

        return unique_node_ids, unique_messages, unique_timestamps


class SumMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(SumMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages):
        """Only keep the last message for each node"""
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []

        for node_id in unique_node_ids:
            unique_messages.append(torch.sum(messages[node_id][0], dim=0))
            unique_timestamps.append(messages[node_id][1][-1])

        unique_messages = torch.stack(unique_messages) if len(unique_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(unique_node_ids) > 0 else []

        return unique_node_ids, unique_messages, unique_timestamps

class ExpLambsMessageAggregator(torch.nn.Module):
    def __init__(self, device, embedding_dimension):
        super(ExpLambsMessageAggregator, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.device = device

    def aggregate(self, node_ids, messages, lambs):
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        to_update_node_ids = []
        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_timestamps.append(messages[node_id][1][-1])
                unique_messages.append(torch.sum(messages[node_id][0].repeat(lambs.shape[0], 1, 1).permute([1, 0, 2]) * torch.exp((messages[node_id][1] - messages[node_id][1][-1]).repeat(lambs.shape[0], 1).T/lambs).unsqueeze(-1), dim=0))

        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps



def get_message_aggregator(aggregator_type, device, embedding_dimension=0):
    if aggregator_type == "mean":
        return MeanMessageAggregator(device=device)
    elif aggregator_type == "sum":
        return SumMessageAggregator(device=device)
    elif aggregator_type == "exp_lambs":
        return ExpLambsMessageAggregator(device=device, embedding_dimension=embedding_dimension)
    else:
        raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
