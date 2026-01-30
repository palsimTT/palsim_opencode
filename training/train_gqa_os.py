"""
Train GQA-OS models for win-rate prediction
Self-contained script - no external model imports required

Usage:
    python train_gqa_os.py --data_dir /path/to/data --weights_dir /path/to/weights --gpu_id 0
"""

import os
import sys
import json
import pickle
import argparse
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost as xgb

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
    """GQA-OS: XGBoost OOF + GQA + MLP
    
    Input:
        - oof_preds: OOF predictions [B, 1] (XGBoost only)
        - original_features: Original features [B, n_features]
    
    Output:
        - Win probability logits [B, 1]
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class OOFXGBoostTrainer:
    def __init__(self, config, seed=42, gpu_id=0):
        self.config = config
        self.seed = seed
        self.gpu_id = gpu_id
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val, y_val):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': self.config.get('max_depth', 10),
            'learning_rate': self.config.get('learning_rate', 0.02),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.7),
            'seed': self.seed,
            'tree_method': 'hist',
            'device': f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu',
            'verbosity': 0
        }
        
        self.model = xgb.train(
            params, dtrain,
            num_boost_round=self.config.get('n_estimators', 1000),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=self.config.get('early_stopping_rounds', 50),
            verbose_eval=False
        )
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)
        return self.model.predict(dtest)
    
    def save(self, path):
        self.model.save_model(str(path / 'xgb_model.json'))
        with open(path / 'xgb_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, path):
        self.model = xgb.Booster()
        self.model.load_model(str(path / 'xgb_model.json'))
        with open(path / 'xgb_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)


def train_oof_xgb_only(X_train, y_train, X_val, X_test, n_splits, seed, base_config, gpu_id=0):
    """Train XGBoost-only OOF"""
    n_train = len(X_train)
    oof_xgb = np.zeros(n_train)
    val_xgb_preds = np.zeros((n_splits, len(X_val)))
    test_xgb_preds = np.zeros((n_splits, len(X_test)))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        xgb_trainer = OOFXGBoostTrainer(base_config['xgb'], seed + fold_id, gpu_id=gpu_id)
        xgb_trainer.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        oof_xgb[val_idx] = xgb_trainer.predict(X_fold_val)
        val_xgb_preds[fold_id] = xgb_trainer.predict(X_val)
        test_xgb_preds[fold_id] = xgb_trainer.predict(X_test)
        print(f"    Fold {fold_id+1}/{n_splits} completed")
    
    oof_train = oof_xgb.reshape(-1, 1)
    oof_val = val_xgb_preds.mean(axis=0).reshape(-1, 1)
    oof_test = test_xgb_preds.mean(axis=0).reshape(-1, 1)
    
    return oof_train, oof_val, oof_test


def train_meta_model(model, oof_train, X_train, y_train, oof_val, X_val, y_val, 
                     config, device, verbose=True):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 2e-4)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss()
    
    oof_train_t = torch.FloatTensor(oof_train).to(device)
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    oof_val_t = torch.FloatTensor(oof_val).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    batch_size = config.get('batch_size', 64)
    epochs = config.get('epochs', 200)
    patience = config.get('patience', 30)
    ortho_weight = config.get('ortho_weight', 0.15)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    n_samples = len(y_train)
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]
            
            optimizer.zero_grad()
            logits = model(oof_train_t[batch_idx], X_train_t[batch_idx])
            bce_loss = criterion(logits, y_train_t[batch_idx])
            ortho_loss = model.get_orthogonality_loss()
            loss = bce_loss + ortho_weight * ortho_loss
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(oof_val_t, X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}: val_loss={val_loss:.4f} (best)")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    
    return model, best_val_loss


def load_data(data_dir):
    train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(data_dir, weights_dir, gpu_id=0, seed=42):
    """Train a single GQA-OS model"""
    set_seed(seed)
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(str(data_dir))
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    n_features = X_train.shape[1]
    
    meta_config = {**DEFAULT_META_CONFIG, 'n_features': n_features}
    base_config = DEFAULT_BASE_CONFIG
    
    print("\n" + "="*60)
    print("Step 1: Training OOF Base Models (XGBoost only)...")
    print("="*60)
    
    oof_train, oof_val, oof_test = train_oof_xgb_only(
        X_train, y_train, X_val, X_test,
        n_splits=5, seed=seed, base_config=base_config, gpu_id=gpu_id
    )
    print(f"  OOF shapes: train={oof_train.shape}, val={oof_val.shape}, test={oof_test.shape}")
    
    np.save(weights_dir / 'oof_train.npy', oof_train)
    
    print("\n" + "="*60)
    print("Step 2: Training Full XGBoost Model for Inference...")
    print("="*60)
    
    xgb_trainer = OOFXGBoostTrainer(base_config['xgb'], seed=seed, gpu_id=gpu_id)
    xgb_trainer.train(X_train, y_train, X_val, y_val)
    xgb_trainer.save(weights_dir)
    print("  XGBoost model saved")
    
    print("\n" + "="*60)
    print("Step 3: Training GQA-OS Meta Model...")
    print("="*60)
    
    model = GQAOS(
        n_features=n_features,
        n_queries=meta_config['n_queries'],
        d_model=meta_config['d_model'],
        n_heads=meta_config['n_heads'],
        fusion_hidden=meta_config['fusion_hidden'],
        fusion_layers=meta_config['fusion_layers']
    ).to(device)
    
    print(f"  GQA-OS parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model, best_loss = train_meta_model(
        model, oof_train, X_train, y_train, oof_val, X_val, y_val,
        meta_config, device, verbose=True
    )
    
    print(f"\n  Training completed. Best val loss: {best_loss:.4f}")
    
    torch.save(model.state_dict(), weights_dir / 'gqa_os_weights.pth')
    print(f"  GQA-OS model saved to: {weights_dir / 'gqa_os_weights.pth'}")
    
    print("\n" + "="*60)
    print("Step 4: Testing...")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        oof_test_t = torch.FloatTensor(oof_test).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        logits = model(oof_test_t, X_test_t)
        y_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    print(f"  Test AUC: {auc:.4f}")
    print(f"  Test Brier Score: {brier:.4f}")
    
    config_to_save = {
        'meta_config': meta_config,
        'base_config': base_config,
        'n_features': n_features,
        'seed': seed,
        'test_auc': float(auc),
        'test_brier': float(brier),
    }
    with open(weights_dir / 'gqa_os_config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    return auc, brier


def main():
    parser = argparse.ArgumentParser(description='Train GQA-OS model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--weights_dir', type=str, required=True, help='Path to save weights')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print(f"# Training GQA-OS Model")
    print(f"# Data: {args.data_dir}")
    print(f"# Weights: {args.weights_dir}")
    print("#"*70)
    
    auc, brier = train_model(args.data_dir, args.weights_dir, args.gpu_id, args.seed)
    
    print("\n" + "="*60)
    print("GQA-OS Training Completed!")
    print("="*60)
    print(f"Weights saved to: {args.weights_dir}")
    print(f"Final Test AUC: {auc:.4f}, Brier: {brier:.4f}")


if __name__ == "__main__":
    main()
