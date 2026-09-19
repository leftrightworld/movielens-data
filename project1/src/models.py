"""MF-BPR and LightGCN implemented from scratch in PyTorch.

Both models learn one embedding per user and per item, score a (user, item)
pair by the dot product, and are trained with the BPR pairwise ranking loss.
The ONLY difference is how the final embeddings are produced:

  MF-BPR  : final embedding = the free embedding table itself.
  LightGCN: final embedding = mean of the embeddings propagated 0..L hops
            over the user-item interaction graph (He et al., SIGIR 2020).

LightGCN propagation is implemented with gather + index_add over the edge
list, which runs on CPU and Apple-Silicon MPS alike (sparse matmul does not).
"""
import torch
import torch.nn as nn


class MFBPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def final_embeddings(self):
        return self.user_emb.weight, self.item_emb.weight

    def ego_embeddings(self, users, pos, neg):
        """Embeddings used for L2 regularization (the free parameters)."""
        return self.user_emb(users), self.item_emb(pos), self.item_emb(neg)


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, n_layers: int,
                 train_edges: torch.Tensor):
        """train_edges: LongTensor (E, 2) of (user_idx, item_idx) train pairs."""
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # Bidirectional edges on the joint (users + items) node index space,
        # with symmetric normalization 1 / sqrt(deg_u * deg_i).
        u, i = train_edges[:, 0], train_edges[:, 1] + n_users
        deg = torch.zeros(n_users + n_items)
        deg.index_add_(0, u, torch.ones(len(u)))
        deg.index_add_(0, i, torch.ones(len(i)))
        w = (deg[u] * deg[i]).clamp(min=1).rsqrt()
        self.register_buffer("src", torch.cat([u, i]))
        self.register_buffer("dst", torch.cat([i, u]))
        self.register_buffer("w", torch.cat([w, w]).unsqueeze(1))

    def propagate(self):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight])
        acc = x
        for _ in range(self.n_layers):
            msg = x[self.src] * self.w
            x = torch.zeros_like(x).index_add_(0, self.dst, msg)
            acc = acc + x
        out = acc / (self.n_layers + 1)          # mean over layers 0..L
        return out[:self.n_users], out[self.n_users:]

    def final_embeddings(self):
        return self.propagate()

    def ego_embeddings(self, users, pos, neg):
        return self.user_emb(users), self.item_emb(pos), self.item_emb(neg)


def bpr_loss(user_e, pos_e, neg_e, ego, reg: float):
    """BPR loss + L2 regularization on the ego (free) embeddings."""
    pos_s = (user_e * pos_e).sum(-1)
    neg_s = (user_e * neg_e).sum(-1)
    loss = -nn.functional.logsigmoid(pos_s - neg_s).mean()
    l2 = sum((e ** 2).sum() for e in ego) / len(user_e)
    return loss + reg * l2
