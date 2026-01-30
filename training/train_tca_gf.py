import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from tca_gf import TCAGF, BEST_CONFIG, create_model


import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    
    with open(os.path.join(data_dir, 'mappings.json'), 'r') as f:
        mappings = json.load(f)
    
    feature_cols = ['ST_t2_x', 'ST_t2_y', 'BP_t2_x', 'BP_t2_y',
                    'ST_t1_x', 'ST_t1_y', 'BP_t1_x', 'BP_t1_y',
                    'ST_t2_mask', 'BP_t2_mask', 'ST_t1_mask', 'BP_t1_mask']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    
    y_st_train = train_df['st_target_idx'].values
    y_bp_train = train_df['bp_target_idx'].values
    y_st_val = val_df['st_target_idx'].values
    y_bp_val = val_df['bp_target_idx'].values
    
    st_classes = len(mappings['st_to_idx'])
    bp_classes = len(mappings['bp_to_idx'])
    
    return {
        'X_train': X_train,
        'y_st_train': y_st_train,
        'y_bp_train': y_bp_train,
        'X_val': X_val,
        'y_st_val': y_st_val,
        'y_bp_val': y_bp_val,
        'scaler': scaler,
        'st_classes': st_classes,
        'bp_classes': bp_classes,
        'mappings': mappings,
    }


def train_model(model, data, device, config, epochs=1000, patience=60, seed=42):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    alpha = config.get('alpha', 0.2)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.LongTensor(data['y_st_train']),
        torch.LongTensor(data['y_bp_train'])
    )
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.LongTensor(data['y_st_val']),
        torch.LongTensor(data['y_bp_val'])
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    st_classes = data['st_classes']
    
    use_amp = device.type == 'cuda'
    grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    print(f"Training with alpha={alpha}, lr={config['lr']}, AMP={use_amp}")
    print(f"Train samples: {len(data['X_train'])}, Val samples: {len(data['X_val'])}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for features, st_target, bp_target in train_loader:
            features = features.to(device)
            st_target = st_target.to(device)
            bp_target = bp_target.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                st_one_hot = F.one_hot(st_target, num_classes=st_classes).float()
                _, bp_logits_hard = model(features, st_override=st_one_hot)
                st_logits_soft, bp_logits_soft = model(features, st_override=None)
                
                loss_st = criterion(st_logits_soft, st_target)
                loss_bp = alpha * criterion(bp_logits_hard, bp_target) + \
                         (1 - alpha) * criterion(bp_logits_soft, bp_target)
                loss = loss_st + loss_bp
            
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, st_target, bp_target in val_loader:
                features = features.to(device)
                st_target = st_target.to(device)
                bp_target = bp_target.to(device)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    st_one_hot = F.one_hot(st_target, num_classes=st_classes).float()
                    _, bp_logits_hard = model(features, st_override=st_one_hot)
                    st_logits_soft, bp_logits_soft = model(features, st_override=None)
                    
                    loss_st = criterion(st_logits_soft, st_target)
                    loss_bp = alpha * criterion(bp_logits_hard, bp_target) + \
                             (1 - alpha) * criterion(bp_logits_soft, bp_target)
                    val_loss += (loss_st + loss_bp).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}: val_loss={avg_val_loss:.4f} (best)")
        else:
            patience_counter += 1
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: val_loss={avg_val_loss:.4f}, patience={patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    
    return model, best_val_loss


def main():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    pipeline_dir = Path(__file__).parent
    data_dir = pipeline_dir / 'data'
    weights_dir = pipeline_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    print("Loading data...")
    data = load_data(str(data_dir))
    print(f"ST classes: {data['st_classes']}, BP classes: {data['bp_classes']}")
    
    model = create_model(
        st_classes=data['st_classes'],
        bp_classes=data['bp_classes'],
        config=BEST_CONFIG
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "="*60)
    print("Training Tactical Generator (seed=42)...")
    print("="*60)
    
    model, best_loss = train_model(model, data, device, BEST_CONFIG)
    
    print(f"\nTraining completed. Best val loss: {best_loss:.4f}")
    
    model_path = weights_dir / 'tca_gf_weights.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to: {model_path}")
    
    scaler_path = weights_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(data['scaler'], f)
    print(f"Scaler saved to: {scaler_path}")
    
    config_path = weights_dir / 'config.json'
    config_to_save = {
        **BEST_CONFIG,
        'st_classes': data['st_classes'],
        'bp_classes': data['bp_classes'],
        'st_to_idx': data['mappings']['st_to_idx'],
        'bp_to_idx': data['mappings']['bp_to_idx'],
        'idx_to_st': {v: k for k, v in data['mappings']['st_to_idx'].items()},
        'idx_to_bp': {v: k for k, v in data['mappings']['bp_to_idx'].items()},
    }
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    print("\n" + "="*60)
    print("All files saved successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
