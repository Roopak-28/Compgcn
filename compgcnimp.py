# PyG & dependencies (for CPU, for Colab, for PyTorch 2.0+)
#!pip install torch torchvision torchaudio
#!pip install torch-geometric

# For some datasets (FB15k-237 is supported out of the box)
#!pip install tqdm numpy



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import RelLinkPredDataset
import os.path as osp
import random

# ========================
# 3. CompGCN Conv Layer
# ========================
class CompGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x,
                 opn='corr', dropout=0.1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.opn = opn
        self.act = act
        self.dropout = torch.nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.rel_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
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
        rel_embed = rel_embed @ self.rel_weight
        return out, rel_embed

# ========================
# 4. CompGCN Model
# ========================
class CompGCNLinkPred(nn.Module):
    def __init__(self, num_nodes, num_rels, emb_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_rels = num_rels
        self.entity_emb = nn.Embedding(num_nodes, emb_dim)
        # +2 for inverse and +1 for self-loop
        self.relation_emb = nn.Embedding(num_rels * 2 + 1, emb_dim)
        self.layers = nn.ModuleList([
            CompGCNConv(emb_dim, emb_dim, num_rels, dropout=dropout, act=F.relu if i < num_layers-1 else lambda x: x)
            for i in range(num_layers)
        ])

    def forward(self, edge_index, edge_type):
        ent = self.entity_emb.weight
        rel = self.relation_emb.weight
        for layer in self.layers:
            ent, rel = layer(ent, edge_index, edge_type, rel)
        return ent, rel

    def decode(self, z, rel, edge_index, edge_type):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        r = rel[edge_type]
        return torch.sum(src * r * dst, dim=-1)

# ========================
# 5. Data Loading
# ========================
def load_dataset():
    DATASET_NAME = 'FB15k-237'
    DATA_PATH = osp.join(osp.expanduser('~'), 'data', 'RelLinkPred')
    print(f"Loading {DATASET_NAME} dataset...")
    dataset = RelLinkPredDataset(root=DATA_PATH, name=DATASET_NAME)
    data = dataset[0]
    return dataset, data

def prepare_data(data, device):
    data = data.to(device)
    print(f"Total nodes: {data.num_nodes}")
    print(f"Relations: {data.num_edge_types}")
    print(f"Train triples: {data.train_edge_index.size(1)}")
    return data

def sample_negatives(edge_index, num_nodes, num_neg_samples, device):
    num_pos = edge_index.size(1)
    neg_src = edge_index[0].repeat_interleave(num_neg_samples)
    neg_dst = torch.randint(0, num_nodes, (len(neg_src),), device=device)
    neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
    return neg_edge_index

# ========================
# 6. Training and Eval
# ========================
def train_model(model, data, optimizer, num_epochs, batch_size, neg_samples, device, log_interval=5):
    criterion = nn.BCEWithLogitsLoss()
    num_train_edges = data.train_edge_index.size(1)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, num_train_edges, batch_size):
            optimizer.zero_grad()
            z, rel = model(data.edge_index, data.edge_type)
            batch_start = i
            batch_end = min(i + batch_size, num_train_edges)
            pos_edge_index = data.train_edge_index[:, batch_start:batch_end]
            pos_edge_type = data.train_edge_type[batch_start:batch_end]
            pos_scores = model.decode(z, rel, pos_edge_index, pos_edge_type)
            pos_labels = torch.ones_like(pos_scores)

            neg_edge_index = sample_negatives(pos_edge_index, data.num_nodes, neg_samples, device)
            neg_edge_type = torch.randint(0, data.num_edge_types, (neg_edge_index.size(1),), device=device)
            neg_scores = model.decode(z, rel, neg_edge_index, neg_edge_type)
            neg_labels = torch.zeros_like(neg_scores)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([pos_labels, neg_labels])
            loss = criterion(scores, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (num_train_edges // batch_size + 1)
        if epoch % log_interval == 0:
            print(f"Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f}")

def evaluate_model(model, data, device):
    model.eval()
    with torch.no_grad():
        z, rel = model(data.edge_index, data.edge_type)
        val_scores = model.decode(z, rel, data.valid_edge_index, data.valid_edge_type)
        acc = (val_scores > 0).float().mean().item()
        print(f"Validation accuracy (positive rate): {acc:.4f}")

# ========================
# 7. Before/After Demo!
# ========================
def get_entity_name(idx, dataset):
    try:
        id2ent = {int(v): k for k, v in dataset.entity2id.items()}
        return id2ent.get(idx, str(idx))
    except:
        return str(idx)

def get_relation_name(idx, dataset):
    try:
        id2rel = {int(v): k for k, v in dataset.relation2id.items()}
        return id2rel.get(idx, str(idx))
    except:
        return str(idx)

def demo_triple_predictions(model, data, dataset, num_examples=5):
    print("\n==== CompGCN Predictions (Before & After) ====\n")
    model.eval()
    with torch.no_grad():
        z, rel = model(data.edge_index, data.edge_type)
        for _ in range(num_examples):
            idx = random.randint(0, data.valid_edge_index.shape[1] - 1)
            h_idx = int(data.valid_edge_index[0, idx])
            t_idx = int(data.valid_edge_index[1, idx])
            r_idx = int(data.valid_edge_type[idx])

            h_name = get_entity_name(h_idx, dataset)
            r_name = get_relation_name(r_idx, dataset)
            t_name = get_entity_name(t_idx, dataset)

            pos_score = model.decode(z, rel,
                                     torch.tensor([[h_idx],[t_idx]], device=z.device),
                                     torch.tensor([r_idx], device=z.device)
                                    ).item()
            pos_pred = "POSSIBLE" if pos_score > 0 else "NOT POSSIBLE"
            print(f"TRUE: ({h_name}, {r_name}, {t_name})\n   Score: {pos_score:.3f} → {pos_pred}")

            fake_t_idx = random.randint(0, data.num_nodes - 1)
            fake_t_name = get_entity_name(fake_t_idx, dataset)
            fake_score = model.decode(z, rel,
                                     torch.tensor([[h_idx],[fake_t_idx]], device=z.device),
                                     torch.tensor([r_idx], device=z.device)
                                    ).item()
            fake_pred = "POSSIBLE" if fake_score > 0 else "NOT POSSIBLE"
            print(f"FAKE: ({h_name}, {r_name}, {fake_t_name})\n   Score: {fake_score:.3f} → {fake_pred}")
            print("-" * 50)

# ========================
# 8. Main Pipeline
# ========================
def main():
    EMBED_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.1
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 20
    BATCH_SIZE = 1024
    NEG_SAMPLES = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset, data = load_dataset()
    data = prepare_data(data, DEVICE)

    model = CompGCNLinkPred(
        num_nodes=data.num_nodes,
        num_rels=data.num_edge_types,
        emb_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_model(model, data, optimizer, NUM_EPOCHS, BATCH_SIZE, NEG_SAMPLES, DEVICE)
    evaluate_model(model, data, DEVICE)

    print("\n" + "="*50)
    print("CompGCN LINK PREDICTION READY")
    print("="*50)

    demo_triple_predictions(model, data, dataset, num_examples=5)

if __name__ == "__main__":
    main()
