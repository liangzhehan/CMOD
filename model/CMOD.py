import logging
import math
import numpy as np
import torch
import torch.nn as nn
from modules.memory import ExpMemory_lambs, Memory_lambs
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module


class NodeMemory(nn.Module):
    def __init__(self, n_nodes, node_features, embedding_dimension, memory_dimension, message_dimension, lambs, device,
                 output, init_lamb=0.5):
        super(NodeMemory, self).__init__()
        self.n_nodes = n_nodes
        self.node_features = node_features
        self.embedding_dimension = embedding_dimension
        self.output = output
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.lambs = torch.Tensor(lambs).to(self.device) * self.output
        self.lamb_len = self.lambs.shape[0]
        raw_message_dimension = self.memory_dimension * self.lamb_len + self.node_features.shape[1]
        self.memory = ExpMemory_lambs(n_nodes=self.n_nodes,
                                      memory_dimension=self.memory_dimension,
                                      lambs=self.lambs,
                                      device=self.device)  # (3, nodes, raw_message_dim)
        self.message_aggregator = get_message_aggregator(aggregator_type="exp_lambs", device=self.device,
                                                         embedding_dimension=memory_dimension)
        self.message_function = get_message_function(module_type="mlp",
                                                     raw_message_dimension=raw_message_dimension,
                                                     message_dimension=self.message_dimension)
        self.memory_updater = get_memory_updater(module_type="exp_lambs",
                                                 memory=self.memory,
                                                 message_dimension=self.message_dimension,
                                                 memory_dimension=self.lambs,
                                                 device=self.device)
        self.exp_embedding = get_embedding_module(module_type="exp_lambs")
        self.iden_embedding = get_embedding_module(module_type="identity")
        self.static_embedding = nn.Embedding(self.n_nodes, self.embedding_dimension)

        self.n_regions = math.ceil(math.sqrt(self.n_nodes))
        self.n_graphs = 1
        self.region_memory = Memory_lambs(n_nodes=self.n_regions,
                                          memory_dimension=self.memory_dimension,
                                          lambs=self.lambs,
                                          device=self.device)
        self.graph_memory = Memory_lambs(n_nodes=self.n_graphs,
                                         memory_dimension=self.memory_dimension,
                                         lambs=self.lambs,
                                         device=self.device)
        self.region_memory_updater = get_memory_updater("exp_lambs", self.region_memory, self.message_dimension,
                                                        self.lambs,
                                                        self.device)
        self.graph_memory_updater = get_memory_updater("exp_lambs", self.graph_memory, self.message_dimension,
                                                       self.lambs,
                                                       self.device)
        self.n_heads = 8
        self.message_per_head = message_dimension // self.n_heads
        self.features_per_head = self.memory_dimension // self.n_heads
        self.message_multi_head = self.message_per_head * self.n_heads
        self.features_multi_head = self.features_per_head * self.n_heads
        self.Q_r = torch.nn.Linear(self.memory_dimension * self.lamb_len, self.features_multi_head, bias=False)
        self.K_r = torch.nn.Linear(self.memory_dimension * self.lamb_len, self.features_multi_head, bias=False)
        self.V_r = torch.nn.Linear(raw_message_dimension, self.message_multi_head, bias=False)
        self.Q_g = torch.nn.Linear(self.memory_dimension * self.lamb_len, self.features_multi_head, bias=False)
        self.K_g = torch.nn.Linear(self.memory_dimension * self.lamb_len, self.features_multi_head, bias=False)
        self.V_g = torch.nn.Linear(self.message_multi_head, self.message_multi_head, bias=False)
        self.ff_r = torch.nn.Sequential(
            torch.nn.Linear(self.message_multi_head, self.message_multi_head, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.message_multi_head, self.message_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.ff_g = torch.nn.Sequential(
            torch.nn.Linear(self.message_multi_head, self.message_multi_head, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.message_multi_head, self.message_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.embedding_transform = torch.nn.Sequential(
            torch.nn.Linear(memory_dimension * self.lamb_len, memory_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.spatial_transform = torch.nn.Sequential(
            torch.nn.Linear(memory_dimension * 3, embedding_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.lamb = nn.Parameter(torch.Tensor([init_lamb]), requires_grad=True)

    def forward(self, source_nodes, target_nodes, edge_timediff, timestamps_batch_torch, now_time,
                begin_time, predict_od):
        # 1. Get node messages from last updated memories
        memory = self.memory.get_memory()
        target_embeddings = self.exp_embedding.compute_embedding(memory=memory,
                                                                 nodes=target_nodes,
                                                                 time_diffs=edge_timediff)  # (nodes, l * memory)
        # Compute node_level messages
        raw_messages = self.get_raw_messages(source_nodes,
                                             target_embeddings,
                                             edge_timediff,
                                             self.node_features[target_nodes],
                                             timestamps_batch_torch)  # (nodes, l * memory + feature)
        unique_nodes, unique_raw_messages, unique_timestamps = self.message_aggregator.aggregate(source_nodes,
                                                                                                 raw_messages,
                                                                                                 self.lambs)  # unique_raw_messages: (nodes, l, raw_message_dim)
        unique_messages = torch.cat(
            [self.message_function.compute_message(unique_raw_messages[:, :, :-1]), unique_raw_messages[:, :, -1:]],
            dim=-1)  # (nodes, l, message_dim)

        # 2. Compute messages for three scales
        region_memory = self.region_memory.get_memory()
        graph_memory = self.graph_memory.get_memory()
        last_update = self.memory.last_update
        time_diffs = - last_update + begin_time
        static_node_embedding = self.exp_embedding.compute_embedding(memory=memory,
                                                                     nodes=list(range(self.n_nodes)),
                                                                     time_diffs=time_diffs)
        region_embedding = self.iden_embedding.compute_embedding(memory=region_memory,
                                                                 nodes=list(range(self.n_regions)),
                                                                 time_diffs=time_diffs).reshape([self.n_regions, -1])
        graph_embedding = self.iden_embedding.compute_embedding(memory=graph_memory,
                                                                nodes=list(range(self.n_graphs)),
                                                                time_diffs=time_diffs).reshape([self.n_graphs, -1])

        # Compute relations between different scales
        A_r = torch.einsum("rhf,nhf->rhn",
                           self.Q_r(region_embedding).reshape([self.n_regions, self.n_heads, self.features_per_head]),
                           self.K_r(static_node_embedding).reshape(
                               [self.n_nodes, self.n_heads, self.features_per_head])) / math.sqrt(
            self.features_per_head)  # r * h * n
        A_g = torch.einsum("ghf,rhf->ghr",
                           self.Q_g(graph_embedding).reshape([self.n_graphs, self.n_heads, self.features_per_head]),
                           self.K_g(region_embedding).reshape(
                               [self.n_regions, self.n_heads, self.features_per_head])) / math.sqrt(
            self.features_per_head)  # g * h * r
        region_messages_mid = torch.einsum("rhn,nlhf->rlhf", torch.softmax(A_r[:, :, unique_nodes], dim=2),
                                           self.V_r(
                                               (unique_raw_messages[:, :, :-1] / unique_raw_messages[:, :,
                                                                                 -1:]).reshape(len(unique_nodes),
                                                                                               self.lamb_len,
                                                                                               -1)).reshape(
                                               [len(unique_nodes), self.lamb_len, self.n_heads,
                                                self.message_per_head])).reshape(
            [self.n_regions, self.lamb_len, self.message_multi_head])
        region_messages = self.ff_r(region_messages_mid)  # (n_regions, lambs, message_dim)
        graph_messages_mid = torch.einsum("ghr,rlhf->glhf", torch.softmax(A_g, dim=2),
                                          self.V_g(region_messages_mid).reshape(
                                              [self.n_regions, self.lamb_len, self.n_heads, self.message_per_head]))
        graph_messages = self.ff_g(graph_messages_mid.reshape([self.n_graphs, self.lamb_len, self.message_multi_head]))

        # 3. Update memories and predict OD matrix
        embeddings = None
        if predict_od:
            updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                         unique_messages,
                                                                                         timestamps=unique_timestamps)
            updated_time_diffs = - updated_last_update + now_time
            recent_node_embeddings = self.exp_embedding.compute_embedding(memory=updated_memory,
                                                                          nodes=list(range(self.n_nodes)),
                                                                          time_diffs=updated_time_diffs)
            updated_region_memory, _ = self.region_memory_updater.get_updated_memory(list(range(self.n_regions)),
                                                                                     region_messages,
                                                                                     timestamps=now_time)
            updated_graph_memory, _ = self.graph_memory_updater.get_updated_memory(list(range(self.n_graphs)),
                                                                                   graph_messages,
                                                                                   timestamps=now_time)
            recent_region_embeddings = self.iden_embedding.compute_embedding(memory=updated_region_memory,
                                                                             nodes=list(range(self.n_regions))).reshape(
                [self.n_regions, -1])
            recent_graph_embeddings = self.iden_embedding.compute_embedding(memory=updated_graph_memory,
                                                                            nodes=list(range(self.n_graphs))).reshape(
                [self.n_graphs, -1])
            r2n = torch.mean(torch.softmax(A_r, dim=0), dim=1)  # r * h * n -> r * n
            g2r = torch.mean(torch.softmax(A_g, dim=0), dim=1)  # g * h * r -> g * r
            region_node_embeddings = torch.mm(r2n.T, recent_region_embeddings)
            graph_node_embeddings = torch.mm(torch.mm(r2n.T, g2r.T), recent_graph_embeddings)
            dynamic_embeddings = torch.cat([recent_node_embeddings, region_node_embeddings, graph_node_embeddings],
                                           dim=0)
            embeddings = self.lamb * self.static_embedding.weight + (1 - self.lamb) * self.spatial_transform(
                self.embedding_transform(
                    dynamic_embeddings).reshape([3, self.n_nodes, self.memory_dimension]).permute([1, 0, 2]).reshape(
                    [self.n_nodes, -1]))

        self.memory_updater.update_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)
        self.region_memory_updater.update_memory(list(range(self.n_regions)), region_messages, timestamps=now_time)
        self.graph_memory_updater.update_memory(list(range(self.n_graphs)), graph_messages, timestamps=now_time)
        return embeddings

    def get_raw_messages(self, source_nodes, target_embeddings, edge_timediff, node_features, edge_times):
        source_message = torch.cat(
            [target_embeddings, node_features, torch.ones([target_embeddings.shape[0], 1]).to(self.device)], dim=1)
        messages = dict()
        unique_nodes = np.unique(source_nodes)
        for node_i in unique_nodes:
            ind = np.arange(source_message.shape[0])[source_nodes == node_i]
            messages[node_i] = [source_message[ind], edge_times[ind]]
        return messages

    def init_memory(self):
        self.memory.__init_memory__()
        self.region_memory.__init_memory__()
        self.graph_memory.__init_memory__()

    def backup_memory(self):
        return [self.memory.backup_memory(), self.region_memory.backup_memory(), self.graph_memory.backup_memory()]

    def restore_memory(self, memory):
        self.memory.restore_memory(memory[0])
        self.region_memory.restore_memory(memory[1])
        self.graph_memory.restore_memory(memory[2])

    def detach_memory(self):
        self.memory.detach_memory()
        self.region_memory.detach_memory()
        self.graph_memory.detach_memory()


class PredictionLayer(nn.Module):
    def __init__(self, embedding_dim, n_nodes):
        super(PredictionLayer, self).__init__()
        self.n_nodes = n_nodes
        self.w = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2), 1),
        )

    def forward(self, embeddings):
        return self.w(torch.cat(
            [embeddings.repeat([1, self.n_nodes]).reshape([self.n_nodes * self.n_nodes, -1]),
             embeddings.repeat([self.n_nodes, 1])],
            dim=1)).reshape([self.n_nodes, self.n_nodes])


class CMOD(nn.Module):
    def __init__(self, device,
                 n_nodes=268, node_features=None,
                 message_dimension=64, memory_dimension=64, lambs=None,
                 output=30):
        super(CMOD, self).__init__()
        if lambs is None:
            lambs = [1]
        self.logger = logging.getLogger(__name__)
        node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        embedding_dimension = memory_dimension * 1
        self.memory = NodeMemory(n_nodes, node_raw_features, embedding_dimension, memory_dimension,
                                 message_dimension, lambs, device,
                                 output, init_lamb=0.1)
        self.predict_od = PredictionLayer(embedding_dimension, n_nodes)

    def compute_od_matrix(self, o_nodes, d_nodes, timestamps_batch_torch,
                          edge_timediff, now_time, begin_time,
                          predict_od=True):
        embeddings = self.memory(o_nodes, d_nodes, edge_timediff,
                                       timestamps_batch_torch, now_time, begin_time,
                                       predict_od)
        od_matrix = None
        if predict_od:
            od_matrix = self.predict_od(embeddings)

        return od_matrix

    def init_memory(self):
        self.memory.init_memory()

    def backup_memory(self):
        return self.memory.backup_memory()

    def restore_memory(self, memories):
        self.memory.restore_memory(memories)

    def detach_memory(self):
        self.memory.detach_memory()
