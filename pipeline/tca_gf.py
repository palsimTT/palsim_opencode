"""
TCA-GF: Temporal Cross-Attention with Gated Fusion(this model is the Tactical Gnerator in the paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCAGF(nn.Module):
    def __init__(self, input_dim=12, st_classes=13, bp_classes=9,
                 d_model=64, n_heads=4, dropout=0.15):
        super().__init__()
        self.st_classes = st_classes
        self.bp_classes = bp_classes
        self.d_model = d_model
        
        # Feature embedding for time steps (split t-2 and t-1)
        self.embed_t2 = nn.Linear(6, d_model)
        self.embed_t1 = nn.Linear(6, d_model)
        
        # Position encoding (Pos)
        self.pos_embed_t2 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed_t1 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Unidirectional Cross-Attention (Uni): t2 attends to t1
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # Temporal fusion
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ST prediction head
        self.st_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, st_classes)
        )
        
        # Attention gating (AttnGate) - for temporal feature fusion
        self.gate_dim = d_model // n_heads
        self.st_key_proj = nn.Linear(1, self.gate_dim)
        self.st_value_proj = nn.Linear(1, self.gate_dim)
        self.query_proj = nn.Linear(d_model, self.gate_dim)
        self.gate_attn = nn.MultiheadAttention(self.gate_dim, 1, dropout=dropout, batch_first=True)
        
        # BP prediction head
        self.bp_head = nn.Sequential(
            nn.Linear(d_model + st_classes, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, bp_classes)
        )
        
    def forward(self, x, st_override=None):
        """
        Args:
            x: [B, 12]
            st_override: hard label (ST ground truth for each sample)
            
        Returns:
            st_logits: logits [B, st_classes]
            bp_logits: logits [B, bp_classes]
        """
        t2_features = torch.cat([x[:, 0:4], x[:, 8:10]], dim=1)  # [B, 6]
        t1_features = torch.cat([x[:, 4:8], x[:, 10:12]], dim=1)  # [B, 6]
        
        t2_embed = self.embed_t2(t2_features).unsqueeze(1) + self.pos_embed_t2
        t1_embed = self.embed_t1(t1_features).unsqueeze(1) + self.pos_embed_t1
        
        # Unidirectional Cross-Attention: t2 attends to t1
        t2_attn, _ = self.cross_attn(t2_embed, t1_embed, t1_embed)
        t2_out = self.norm(t2_embed + t2_attn).squeeze(1)
        t1_out = t1_embed.squeeze(1)
        
        temporal_repr = self.temporal_fusion(torch.cat([t2_out, t1_out], dim=1))
        
        st_logits = self.st_head(temporal_repr)
        st_probs = F.softmax(st_logits, dim=1)
        st_input = st_override if st_override is not None else st_probs
        
        st_tokens = st_input.unsqueeze(-1)
        st_keys = self.st_key_proj(st_tokens)
        st_values = self.st_value_proj(st_tokens)
        query = self.query_proj(temporal_repr).unsqueeze(1)
        _, attn_weights = self.gate_attn(query, st_keys, st_values)
        gate_values = attn_weights.squeeze(1)
        gated_st = st_input * gate_values
        
        bp_logits = self.bp_head(torch.cat([temporal_repr, gated_st], dim=1))
        
        return st_logits, bp_logits
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            st_logits, bp_logits = self.forward(x)
            st_probs = F.softmax(st_logits, dim=1)
            bp_probs = F.softmax(bp_logits, dim=1)
        return st_probs, bp_probs


# Best hyperparameters found through grid search
BEST_CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'dropout': 0.15,
    'lr': 5e-05,
    'alpha': 0.2,
}


def create_model(st_classes=13, bp_classes=9, config=None):
    if config is None:
        config = BEST_CONFIG
    
    return TCAGF(
        input_dim=12,
        st_classes=st_classes,
        bp_classes=bp_classes,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        dropout=config['dropout']
    )
