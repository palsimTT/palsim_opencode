"""GQA-OS: Grouped Query Attention with OOF Stacking (Win-rate Predictor)

Updated version: 
- Removed TabR, using XGBoost-only OOF
- Replaced MQA with GQA (Grouped Query Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) - Multiple query heads share KV heads"""
    def __init__(self, input_dim, n_queries=8, d_model=64, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.input_dim = input_dim
        self.n_queries = n_queries
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_head_dim = d_model // n_kv_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // n_kv_heads
        
        init_queries = torch.zeros(1, n_queries, d_model)
        nn.init.orthogonal_(init_queries[0])
        self.learnable_queries = nn.Parameter(init_queries)
        
        self.feature_embed = nn.Linear(1, d_model)
        self.position_emb = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.value_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        B = x.size(0)
        x_seq = x.unsqueeze(-1)
        x_embed = self.feature_embed(x_seq) + self.position_emb
        
        Q = self.learnable_queries.expand(B, -1, -1)
        K = self.key_proj(x_embed)
        V = self.value_proj(x_embed)
        
        Q = Q.view(B, self.n_queries, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, self.input_dim, self.n_kv_heads, self.head_dim)
        V = V.view(B, self.input_dim, self.n_kv_heads, self.head_dim)
        K = K.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)
        V = V.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, self.n_queries, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output.view(B, -1), attn_weights
    
    def get_orthogonality_loss(self):
        Q = self.learnable_queries.squeeze(0)
        Q_norm = F.normalize(Q, dim=1)
        similarity = torch.mm(Q_norm, Q_norm.t())
        eye = torch.eye(self.n_queries, device=Q.device)
        off_diag = similarity * (1 - eye)
        return off_diag.pow(2).sum()


class GQAOS(nn.Module):
    """GQA-OS: XGBoost OOF + GQA + MLP (without TabR)
    
    Input:
        - oof_preds: OOF predictions [B, 1] (XGBoost only)
        - original_features: Original features [B, n_features]
    
    Output:
        - Win probability [B, 1]
    """
    def __init__(self, n_features=16, n_queries=8, d_model=64, n_heads=8, 
                 fusion_hidden=64, fusion_layers=3, n_kv_heads=2):
        super().__init__()
        self.attention = GroupedQueryAttention(n_features, n_queries, d_model, n_heads, n_kv_heads)
        
        attn_output_dim = n_queries * d_model
        fusion_input_dim = 1 + attn_output_dim 
        
        layers = []
        in_dim = fusion_input_dim
        for i in range(fusion_layers):
            out_dim = fusion_hidden // (2 ** i) if i < fusion_layers - 1 else fusion_hidden // (2 ** (fusion_layers - 1))
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(0.3 if i == 0 else 0.2)
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.fusion = nn.Sequential(*layers)
        
    def forward(self, oof_preds, original_features):
        if oof_preds.dim() == 1:
            oof_preds = oof_preds.unsqueeze(1)
        attn_output, _ = self.attention(original_features)
        combined = torch.cat([oof_preds, attn_output], dim=1)
        return self.fusion(combined)
    
    def get_orthogonality_loss(self):
        return self.attention.get_orthogonality_loss()


DEFAULT_META_CONFIG = {
    'n_queries': 8,
    'd_model': 64,
    'n_heads': 8,
    'fusion_hidden': 64,
    'fusion_layers': 3,
    'learning_rate': 1e-3,
    'weight_decay': 2e-4,
    'batch_size': 64,
    'epochs': 200,
    'patience': 30,
    'ortho_weight': 0.15
}

DEFAULT_BASE_CONFIG = {
    'xgb': {
        'max_depth': 10,
        'learning_rate': 0.02,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'early_stopping_rounds': 50
    }
}
