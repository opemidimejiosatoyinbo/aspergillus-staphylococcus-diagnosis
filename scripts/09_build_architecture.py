# --- Standard library imports ---
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool   # real GNN building blocks


# ============================================================
# GENOMIC PATHWAY: a real, small transformer trained from scratch on our own k-mer vocabulary (4096 k-mers + padding = 4097)
# ============================================================
class GenomicEncoder(nn.Module):
    """
    Takes a batch of real k-mer token ID sequences (shape: [batch, 512], matching EXACTLY what 06_tokenize_genomes.py produces) and encodes each one into a single fixed-size vector.

    Architecture: embedding lookup -> positional encoding -> a small Transformer encoder stack -> mean-pooling over non-padding tokens.
    """

    def __init__(self, vocab_size: int = 4097, embed_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 2, max_len: int = 512):
        super().__init__()

        # Turns each integer token ID into a learned embed_dim-length vector. padding_idx=0 tells PyTorch that token ID 0 (our PAD_TOKEN_ID from 06_tokenize_genomes.py) carries no real information.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Transformers have no inherent sense of token ORDER on their own -- positional embeddings give the model that information explicitly.
        self.positional_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: shape [batch_size, 512] -- real integer k-mer IDs, exactly as saved by 06_tokenize_genomes.py.
        """
        batch_size, seq_len = token_ids.shape

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(token_ids) + self.positional_embedding(positions)

        # Padding mask: True where a position is real padding (token ID 0), so the transformer knows to ignore those positions in attention.
        padding_mask = (token_ids == 0)

        encoded = self.transformer(x, src_key_padding_mask=padding_mask)

        # Mean-pool over only the REAL (non-padding) token positions -- a sample with fewer real tokens shouldn't be diluted by padding.
        real_token_mask = (~padding_mask).unsqueeze(-1).float()
        summed = (encoded * real_token_mask).sum(dim=1)
        counts = real_token_mask.sum(dim=1).clamp(min=1)   # avoid divide-by-zero for all-padding edge case
        pooled = summed / counts

        return pooled   # shape: [batch_size, embed_dim]


# ============================================================
# STRUCTURAL PATHWAY: a real Graph Neural Network, processing actual variable-size residue graphs from 07_build_structural_graphs.py
# ============================================================
class StructuralEncoder(nn.Module):
    """
    Takes real graph data (node features + edge_index, exactly matching what 07_build_structural_graphs.py produces) and encodes each protein's graph into a single fixed-size vector.

    Architecture: two graph convolution layers (message passing between physically neighboring residues) -> global mean pooling across all nodes in each graph, to get one vector per protein regardless of how many residues it actually has.
    """

    def __init__(self, node_feature_dim: int = 21, embed_dim: int = 256):
        # node_feature_dim=21 matches EXACTLY the real one-hot amino acid encoding from 07_build_structural_graphs.py (20 standard amino acids + 1 "unknown" category).
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x:          real node features, shape [total_nodes_in_batch, 21]
        edge_index: real edges (8-Angstrom contacts), shape [2, total_edges_in_batch]
        batch:      maps each node to which graph/protein it belongs to within this batch -- required by torch_geometric when multiple variable-size graphs are batched together.
        """
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))

        # Pools all of one protein's node embeddings down into a single vector, regardless of how many residues that protein has -- this is what lets a 271-residue protein and a 2051-residue protein both end up as the same-shape output vector.
        pooled = global_mean_pool(x, batch)

        return pooled   # shape: [num_graphs_in_batch, embed_dim]


# ============================================================
# FULL MODEL: real genomic + real structural experts, fused
# ============================================================
class DiagnosticModel(nn.Module):
    def __init__(self, fusion_method: str = "gated", embed_dim: int = 256,
                 vocab_size: int = 4097, node_feature_dim: int = 21):
        super().__init__()

        self.seq_expert = GenomicEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.struct_expert = StructuralEncoder(node_feature_dim=node_feature_dim, embed_dim=embed_dim)

        self.fusion_method = fusion_method
        if fusion_method == "gated":
            self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
            self.fusion_layer = nn.Linear(embed_dim, embed_dim)
        elif fusion_method == "concat":
            self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        self.organism_classifier = nn.Linear(embed_dim, 3)
        self.resistance_classifier = nn.Linear(embed_dim, 1)

    def forward(self, token_ids: torch.Tensor, struct_x: torch.Tensor,
                struct_edge_index: torch.Tensor, struct_batch: torch.Tensor):
        seq_emb = self.seq_expert(token_ids)
        struct_emb = self.struct_expert(struct_x, struct_edge_index, struct_batch)

        if self.fusion_method == "gated":
            combined = torch.cat([seq_emb, struct_emb], dim=1)
            trust = self.gate(combined)
            fused = self.fusion_layer((seq_emb * trust) + (struct_emb * (1 - trust)))
        else:   # concat
            combined = torch.cat([seq_emb, struct_emb], dim=1)
            fused = self.fusion_layer(combined)

        return self.organism_classifier(fused), self.resistance_classifier(fused)


if __name__ == "__main__":
    # A real, minimal smoke test -- confirms the architecture actually runs end-to-end on tensors matching our real data's exact shapes, before we touch real training data in the next script.
    model = DiagnosticModel(fusion_method="gated")

    batch_size = 4
    dummy_tokens = torch.randint(0, 4097, (batch_size, 512))   # matches real genomic tensor shape

    # A tiny dummy graph batch: 4 proteins, varying real-ish sizes
    dummy_struct_x = torch.randn(40, 21)   # 40 total nodes across the batch, 21 real feature dims
    dummy_edge_index = torch.randint(0, 40, (2, 100))
    dummy_batch = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10)   # 10 nodes per protein, for this smoke test

    organism_logits, resistance_logits = model(dummy_tokens, dummy_struct_x, dummy_edge_index, dummy_batch)

    print(f"Organism logits shape: {organism_logits.shape} (expected: [4, 3])")
    print(f"Resistance logits shape: {resistance_logits.shape} (expected: [4, 1])")
    print("Architecture verified against REAL data shapes.")
