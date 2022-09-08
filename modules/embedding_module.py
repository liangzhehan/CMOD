from torch import nn


class EmbeddingModule(nn.Module):
    def __init__(self):
        super(EmbeddingModule, self).__init__()

    def compute_embedding(self, memory, nodes, time_diffs=None):
        pass


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, nodes, time_diffs=None):
        return memory[nodes, :]


class ExpLambsEmbedding(EmbeddingModule):
    def __init__(self):
        super(ExpLambsEmbedding, self).__init__()

    def compute_embedding(self, memory, nodes, time_diffs=None):
        embeddings = (memory[nodes, :, :-1] / memory[nodes, :, -1:]).reshape([len(nodes), -1])
        return embeddings


def get_embedding_module(module_type):
    if module_type == "identity":
        return IdentityEmbedding()
    elif module_type == "exp_lambs":
        return ExpLambsEmbedding()
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
