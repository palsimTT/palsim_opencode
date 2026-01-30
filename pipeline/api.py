"""
PaLSim API
1. predict_distribution: input:context, output:st_probs, bp_probs, joint
2. predict_winrate: input:joint, context, output:winrate_matrix, expected_winrate
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from ast import literal_eval

import torch
import torch.nn.functional as F
import xgboost as xgb

try:
    from .tca_gf import create_model
    from .ipf_solver import IPFSolver, load_prior
    from .gqa_os import GQAOS
except ImportError:
    from tca_gf import create_model
    from ipf_solver import IPFSolver, load_prior
    from gqa_os import GQAOS


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

SUPPORTED_PLAYERS = ['default', 'm', 'w']


class PaLSim:
    def __init__(self, weights_dir: Optional[str] = None, device: Optional[str] = None, 
                 player: str = 'default'):
        if player not in SUPPORTED_PLAYERS:
            raise ValueError(f"Unsupported player: {player}. Supported: {SUPPORTED_PLAYERS}")
        
        self.player = player
        pipeline_dir = Path(__file__).parent
        project_dir = pipeline_dir.parent
        
        if weights_dir is None:
            if player == 'w':
                weights_dir = project_dir / 'weights_w'
            else:  # 'default' or 'm' use the same weights
                weights_dir = project_dir / 'weights_m'
        else:
            weights_dir = Path(weights_dir)
        
        # Data directory (shared for all players)
        self.data_dir = project_dir / 'data_format'
        
        # Prior joint distribution based on player
        if player == 'w':
            self.prior_file = 'st_bp_distribution_prior_w.json'
        else:
            self.prior_file = 'st_bp_distribution_prior_m.json'
        
        self.device = get_device(device)
        self.weights_dir = weights_dir
        
        self._load_config()
        self._load_tcagf_model()
        self._load_gqaos_model()
        self._load_prior()
    
    def _load_config(self):
        config_path = self.weights_dir / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.st_to_idx = self.config['st_to_idx']
        self.bp_to_idx = self.config['bp_to_idx']
        self.idx_to_st = {int(v): k for k, v in self.st_to_idx.items()}
        self.idx_to_bp = {int(v): k for k, v in self.bp_to_idx.items()}
        self.st_classes = self.config['st_classes']
        self.bp_classes = self.config['bp_classes']
        
        self.st_labels = [self.idx_to_st[i] for i in range(self.st_classes)]
        self.bp_labels = [self.idx_to_bp[i] for i in range(self.bp_classes)]
        
        encoding_path = self.data_dir / 'encoding_format.json'
        with open(encoding_path, 'r') as f:
            encoding = json.load(f)
        
        self.st_to_coords = {name: literal_eval(coords) for name, coords in encoding['ST'].items()}
        self.bp_to_coords = {name: literal_eval(coords) for name, coords in encoding['BP'].items()}
    
    def _load_tcagf_model(self):
        """load tcagf model(tcagf model is Tactical Generator)"""
        self.tcagf_model = create_model(
            st_classes=self.st_classes,
            bp_classes=self.bp_classes,
            config=self.config
        )
        self.tcagf_model.load_state_dict(
            torch.load(self.weights_dir / 'tca_gf_weights.pth', map_location=self.device)
        )
        self.tcagf_model.to(self.device)
        self.tcagf_model.eval()
        
        with open(self.weights_dir / 'scaler.pkl', 'rb') as f:
            self.tcagf_scaler = pickle.load(f)
    
    def _load_gqaos_model(self):
        """load gqaos model (GQA-OS is Win-rate Predictor, XGBoost-only OOF + GQA)"""
        gqa_config_path = self.weights_dir / 'gqa_os_config.json'
        with open(gqa_config_path, 'r') as f:
            gqa_config = json.load(f)
        
        self.n_features = gqa_config['n_features']
        meta_config = gqa_config['meta_config']
        
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(self.weights_dir / 'xgb_model.json'))
        with open(self.weights_dir / 'xgb_scaler.pkl', 'rb') as f:
            self.xgb_scaler = pickle.load(f)
        
        self.gqaos_model = GQAOS(
            n_features=self.n_features,
            n_queries=meta_config.get('n_queries', 8),
            d_model=meta_config.get('d_model', 64),
            n_heads=meta_config.get('n_heads', 8),
            fusion_hidden=meta_config.get('fusion_hidden', 64),
            fusion_layers=meta_config.get('fusion_layers', 3)
        ).to(self.device)
        self.gqaos_model.load_state_dict(
            torch.load(self.weights_dir / 'gqa_os_weights.pth', map_location=self.device)
        )
        self.gqaos_model.eval()
    
    def _load_prior(self):
        """load prior distribution"""
        prior_path = self.data_dir / self.prior_file
        if prior_path.exists():
            prior_matrix, prior_st_labels, prior_bp_labels = load_prior(str(prior_path))
            aligned = np.zeros((self.st_classes, self.bp_classes))
            for i, st in enumerate(self.st_labels):
                for j, bp in enumerate(self.bp_labels):
                    if st in prior_st_labels and bp in prior_bp_labels:
                        src_i = prior_st_labels.index(st)
                        src_j = prior_bp_labels.index(bp)
                        aligned[i, j] = prior_matrix[src_i, src_j]
            aligned = aligned + 1e-8
            self.prior_matrix = aligned / aligned.sum()
        else:
            self.prior_matrix = np.ones((self.st_classes, self.bp_classes)) / (self.st_classes * self.bp_classes)
    
    def _encode_context(self, st_t2: Optional[str], bp_t2: Optional[str],
                        st_t1: Optional[str], bp_t1: Optional[str]) -> np.ndarray:
        """encode context"""
        features = np.zeros(12)
        
        if st_t2 and st_t2 in self.st_to_coords:
            coords = self.st_to_coords[st_t2]
            features[0], features[1] = coords[0], coords[1]
            features[8] = 1.0
        
        if bp_t2 and bp_t2 in self.bp_to_coords:
            coords = self.bp_to_coords[bp_t2]
            features[2], features[3] = coords[0], coords[1]
            features[9] = 1.0
        
        if st_t1 and st_t1 in self.st_to_coords:
            coords = self.st_to_coords[st_t1]
            features[4], features[5] = coords[0], coords[1]
            features[10] = 1.0
        
        if bp_t1 and bp_t1 in self.bp_to_coords:
            coords = self.bp_to_coords[bp_t1]
            features[6], features[7] = coords[0], coords[1]
            features[11] = 1.0
        
        return features
    
    def _get_oof_predictions(self, X: np.ndarray) -> np.ndarray:
        """get XGBoost OOF predictions (no TabR)"""
        X_xgb = self.xgb_scaler.transform(X)
        dtest = xgb.DMatrix(X_xgb)
        xgb_pred = self.xgb_model.predict(dtest)
        
        if len(xgb_pred.shape) == 0:
            xgb_pred = np.array([xgb_pred])
        
        return xgb_pred.reshape(-1, 1)
    
    def _predict_single_winrate(self, features: np.ndarray) -> float:
        """predict single sample winrate"""
        features = features.reshape(1, -1)
        oof_preds = self._get_oof_predictions(features)
        
        with torch.no_grad():
            oof_t = torch.FloatTensor(oof_preds).to(self.device)
            X_t = torch.FloatTensor(features).to(self.device)
            logits = self.gqaos_model(oof_t, X_t)
            prob = torch.sigmoid(logits).squeeze().cpu().item()
        
        return prob
    
    def get_valid_strategies(self) -> Dict[str, List[str]]:
        """get valid strategies list"""
        return {
            'st': list(self.st_to_coords.keys()),
            'bp': list(self.bp_to_coords.keys())
        }
    
    def predict_distribution(
        self,
        st_t2: Optional[str] = None,
        bp_t2: Optional[str] = None,
        st_t1: Optional[str] = None,
        bp_t1: Optional[str] = None
    ) -> Dict:
        """predict ST/BP probability distribution and joint distribution
        Returns:
            dict: {
                'st_probs': List[float],     # ST probability distribution (length=st_classes)
                'bp_probs': List[float],     # BP probability distribution (length=bp_classes)
                'joint': List[List[float]],  # joint distribution (st_classes x bp_classes)
                'st_labels': List[str],      # ST labels list
                'bp_labels': List[str],      # BP labels list
            }
        """
        features = self._encode_context(st_t2, bp_t2, st_t1, bp_t1)
        features_scaled = self.tcagf_scaler.transform(features.reshape(1, -1))
        input_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        with torch.no_grad():
            st_logits, bp_logits = self.tcagf_model(input_tensor)
            st_probs = F.softmax(st_logits, dim=1)[0].cpu().numpy()
            bp_probs = F.softmax(bp_logits, dim=1)[0].cpu().numpy()
        
        solver = IPFSolver(self.prior_matrix, self.st_labels, self.bp_labels)
        joint = solver.fit(st_probs, bp_probs)
        
        return {
            'st_probs': st_probs.tolist(),
            'bp_probs': bp_probs.tolist(),
            'joint': joint.tolist(),
            'st_labels': self.st_labels,
            'bp_labels': self.bp_labels,
        }
    
    def predict_winrate(
        self,
        joint: Union[List[List[float]], np.ndarray],
        st_t2: Optional[str] = None,
        bp_t2: Optional[str] = None,
        st_t1: Optional[str] = None,
        bp_t1: Optional[str] = None
    ) -> Dict:
        """predict winrate matrix and expected winrate
        Returns:
            dict: {
                'winrate_matrix': List[List[float]],  # winrate matrix (st_classes x bp_classes)
                'expected_winrate': float,            # expected winrate (weighted by joint distribution)
                'st_labels': List[str],               # ST labels list
                'bp_labels': List[str],               # BP labels list
            }
        """
        context_features = self._encode_context(st_t2, bp_t2, st_t1, bp_t1)
        
        winrate_matrix = np.zeros((self.st_classes, self.bp_classes))
        for i, st in enumerate(self.st_labels):
            for j, bp in enumerate(self.bp_labels):
                st_coords = self.st_to_coords.get(st, (0, 0))
                bp_coords = self.bp_to_coords.get(bp, (0, 0))
                
                full_features = np.concatenate([
                    context_features,
                    np.array([st_coords[0], st_coords[1]]),
                    np.array([bp_coords[0], bp_coords[1]])
                ])
                
                winrate_matrix[i, j] = self._predict_single_winrate(full_features)
        
        joint = np.array(joint)
        if joint.sum() > 0:
            joint_normalized = joint / joint.sum()
        else:
            joint_normalized = joint
        
        expected_winrate = float(np.sum(joint_normalized * winrate_matrix))
        
        return {
            'winrate_matrix': winrate_matrix.tolist(),
            'expected_winrate': expected_winrate,
            'st_labels': self.st_labels,
            'bp_labels': self.bp_labels,
        }
    
    def predict(
        self,
        st_t2: Optional[str] = None,
        bp_t2: Optional[str] = None,
        st_t1: Optional[str] = None,
        bp_t1: Optional[str] = None
    ) -> Dict:
        """end-to-end prediction
        Returns:
            dict: {
                'st_probs': List[float],
                'bp_probs': List[float],
                'joint': List[List[float]],
                'winrate_matrix': List[List[float]],
                'expected_winrate': float,
                'st_labels': List[str],
                'bp_labels': List[str],
            }
        """
        dist_result = self.predict_distribution(st_t2, bp_t2, st_t1, bp_t1)
        wr_result = self.predict_winrate(dist_result['joint'], st_t2, bp_t2, st_t1, bp_t1)
        
        return {
            'st_probs': dist_result['st_probs'],
            'bp_probs': dist_result['bp_probs'],
            'joint': dist_result['joint'],
            'winrate_matrix': wr_result['winrate_matrix'],
            'expected_winrate': wr_result['expected_winrate'],
            'st_labels': dist_result['st_labels'],
            'bp_labels': dist_result['bp_labels'],
        }


def demo():
    print("demo api")

if __name__ == "__main__":
    demo()
