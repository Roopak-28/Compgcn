import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from tqdm import tqdm
import numpy as np

# ---- CompGCNConv Implementation START ----
class CompGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x,
                 opn='corr', dropout=0.1, bias=True):
        super(CompGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.opn = opn
        self.act = act
        self.dropout = torch.nn.Dropout(dropout)
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.rel_weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.rel_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def comp(self, h, r, opn):
        if opn == 'corr':
            return self.circular_correlation(h, r)
        elif opn == 'sub':
            return h - r
        elif opn == 'mult':
            return h * r
        else:
            raise NotImplementedError

    def circular_correlation(self, h, r):
        fft_h = torch.fft.fft(h, dim=-1)
        fft_r = torch.fft.fft(r, dim=-1)
        conj_fft_h = torch.conj(fft_h)
        corr = torch.fft.ifft(conj_fft_h * fft_r, dim=-1).real
        return corr

    def forward(self, x, edge_index, edge_type, rel_embed):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        self_loop_edge = torch.arange(0, num_nodes, dtype=torch.long, device=x.device)
        self_loop_edge = self_loop_edge.unsqueeze(0).repeat(2, 1)
        self_loop_type = torch.full((num_nodes,), self.num_rels * 2, dtype=torch.long, device=x.device)

        edge_index = torch.cat([edge_index, self_loop_edge], dim=1)
        edge_type = torch.cat([edge_type, self_loop_type], dim=0)

        rel_embed = torch.cat([rel_embed, torch.zeros(1, rel_embed.size(1), device=x.device)], dim=0)

        h = x[edge_index[0]]
        r = rel_embed[edge_type]
        msg = self.comp(h, r, self.opn)
        out = torch.zeros_like(x)
        out = out.index_add(0, edge_index[1], msg)
        out = out @ self.weight
        if self.bias is not None:
            out = out + self.bias
        out = self.act(out)
        out = self.dropout(out)
        # Update relation embeddings
        rel_embed = rel_embed @ self.rel_weight
        return out, rel_embed

# ---- CompGCNConv Implementation END ----

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load AIFB dataset for demonstration (can use 'am', 'mutag', 'bgs' as well)
dataset = Entities(root='./data/AIFB', name='aifb')
data = dataset[0].to(device)

# Determine entity and relation counts from the data object
num_entities = data.num_nodes
num_relations = int(data.edge_type.max()) + 1

# CompGCN Model
class CompGCNNet(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim=200):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_relations = num_relations

        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations * 2 + 1, emb_dim)  # include inverses and self-loop

        self.conv1 = CompGCNConv(emb_dim, emb_dim, num_relations)
        self.conv2 = CompGCNConv(emb_dim, emb_dim, num_relations)

    def forward(self, x, edge_index, edge_type):
        ent = self.entity_emb(x)
        rel = self.relation_emb.weight
        ent, rel = self.conv1(ent, edge_index, edge_type, rel)
        ent, rel = self.conv2(ent, edge_index, edge_type, rel)
        return ent, rel

    def score(self, h, r, t):
        # Simple DistMult score
        return torch.sum(h * r * t, dim=-1)

# Instantiate model
model = CompGCNNet(num_entities, num_relations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 5  # Increase this for real training
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward
    x = torch.arange(num_entities, device=device)
    entity_embs, relation_embs = model(x, data.edge_index, data.edge_type)

    # Get positive samples
    pos_h = data.edge_index[0]
    pos_t = data.edge_index[1]
    pos_r = data.edge_type

    # Negative sampling (random tail replacement)
    neg_t = torch.randint(0, num_entities, pos_t.size(), device=device)

    # Get embeddings
    h_emb = entity_embs[pos_h]
    r_emb = relation_embs[pos_r]
    t_emb = entity_embs[pos_t]
    neg_t_emb = entity_embs[neg_t]

    # Scores
    pos_scores = model.score(h_emb, r_emb, t_emb)
    neg_scores = model.score(h_emb, r_emb, neg_t_emb)

    # Loss: Margin Ranking Loss
    margin = 1.0
    target = torch.ones_like(pos_scores)
    loss = F.margin_ranking_loss(pos_scores, neg_scores, target, margin=margin)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ----
# Evaluation: Analyze positive/negative score distributions & pick a threshold
model.eval()
with torch.no_grad():
    # Recompute embeddings
    x = torch.arange(num_entities, device=device)
    entity_embs, relation_embs = model(x, data.edge_index, data.edge_type)
    
    # Collect scores for positive and negative triples
    pos_score_list = []
    neg_score_list = []
    n = 100  # how many examples to sample

    for i in range(n):
        idx = torch.randint(0, pos_h.size(0), (1,)).item()
        h_idx = pos_h[idx].item()
        r_idx = pos_r[idx].item()
        t_idx = pos_t[idx].item()
        # Positive
        pos_score = model.score(
            entity_embs[h_idx].unsqueeze(0),
            relation_embs[r_idx].unsqueeze(0),
            entity_embs[t_idx].unsqueeze(0),
        ).item()
        pos_score_list.append(pos_score)
        # Negative (corrupt tail)
        random_t_idx = torch.randint(0, entity_embs.size(0), (1,)).item()
        neg_score = model.score(
            entity_embs[h_idx].unsqueeze(0),
            relation_embs[r_idx].unsqueeze(0),
            entity_embs[random_t_idx].unsqueeze(0),
        ).item()
        neg_score_list.append(neg_score)

    # Compute a threshold (midpoint between means)
    pos_mean = np.mean(pos_score_list)
    neg_mean = np.mean(neg_score_list)
    threshold = (pos_mean + neg_mean) / 2
    print(f"Mean positive score: {pos_mean:.2f}")
    print(f"Mean negative score: {neg_mean:.2f}")
    print(f"Suggested threshold: {threshold:.2f}")

    # Print out prediction results for sampled triples
    print("\nResults for 10 random POSITIVE triples:")
    for i in range(10):
        idx = torch.randint(0, pos_h.size(0), (1,)).item()
        h_idx = pos_h[idx].item()
        r_idx = pos_r[idx].item()
        t_idx = pos_t[idx].item()
        score = model.score(
            entity_embs[h_idx].unsqueeze(0),
            relation_embs[r_idx].unsqueeze(0),
            entity_embs[t_idx].unsqueeze(0),
        ).item()
        correct = "CORRECT" if score > threshold else "INCORRECT"
        print(f"Triple ({h_idx}, {r_idx}, {t_idx}) — Score: {score:.2f} — Prediction: {correct}")

    print("\nResults for 10 random NEGATIVE triples:")
    for i in range(10):
        idx = torch.randint(0, pos_h.size(0), (1,)).item()
        h_idx = pos_h[idx].item()
        r_idx = pos_r[idx].item()
        random_t_idx = torch.randint(0, entity_embs.size(0), (1,)).item()
        score = model.score(
            entity_embs[h_idx].unsqueeze(0),
            relation_embs[r_idx].unsqueeze(0),
            entity_embs[random_t_idx].unsqueeze(0),
        ).item()
        correct = "INCORRECT" if score > threshold else "CORRECT"
        print(f"Fake triple ({h_idx}, {r_idx}, {random_t_idx}) — Score: {score:.2f} — Prediction: {correct}")
