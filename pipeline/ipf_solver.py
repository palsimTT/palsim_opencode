"""
IPF Solver: Iterative Proportional Fitting:
Convert marginal distributions to joint distribution
"""

import json
import numpy as np


class IPFSolver:
    def __init__(self, prior_matrix, row_labels, col_labels):
        """
        Args:
            prior_matrix: [n_st, n_bp] prior joint distribution
            row_labels: ST marginal distributions (length n_st)
            col_labels: BP marginal distributions (length n_bp)
        """
        self.P = np.array(prior_matrix, dtype=float) # current distributions during iteration
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.num_rows, self.num_cols = self.P.shape
        
    def fit(self, target_row_marginals, target_col_marginals, tol=1e-6, max_iter=1000):
        """
        Args:
            target_row_marginals: (sum=1, length n_st)
            target_col_marginals: (sum=1, length n_bp)
            tol: tolerance for convergence
            max_iter: maximum number of iterations
        Returns:
            P_posterior: posterior joint distributions of st and bp
        """
        r = np.array(target_row_marginals)
        c = np.array(target_col_marginals)
        
        if not np.isclose(r.sum(), 1.0):
            r = r / r.sum()
        if not np.isclose(c.sum(), 1.0):
            c = c / c.sum()
            
        P_curr = self.P.copy()
        if P_curr.sum() > 0:
            P_curr = P_curr / P_curr.sum()
            
        for k in range(max_iter):
            # row
            row_sums = P_curr.sum(axis=1)
            row_scaling = np.zeros_like(row_sums)
            mask = row_sums > 0 
            row_scaling[mask] = r[mask] / row_sums[mask]
            P_curr = P_curr * row_scaling[:, np.newaxis]
            
            # column
            col_sums = P_curr.sum(axis=0)
            col_scaling = np.zeros_like(col_sums)
            mask = col_sums > 0 
            col_scaling[mask] = c[mask] / col_sums[mask]
            P_curr = P_curr * col_scaling[np.newaxis, :]
            
            # check convergence（check if the difference between the current distribution and the target distribution is smaller than the tolerance）
            curr_row_sums = P_curr.sum(axis=1)
            curr_col_sums = P_curr.sum(axis=0)
            
            delta_r = np.max(np.abs(curr_row_sums - r))
            delta_c = np.max(np.abs(curr_col_sums - c))
            delta = max(delta_r, delta_c)
            
            if delta < tol:
                break
                
        return P_curr


def load_prior(json_path):
    """ load prior matrix from JSON file """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # keep st and bp label in same order for consistent indexing
    st_labels = sorted(data.keys())
    bp_labels = sorted(data[st_labels[0]].keys())
    
    matrix = np.zeros((len(st_labels), len(bp_labels)))
    for i, st in enumerate(st_labels):
        for j, bp in enumerate(bp_labels):
            matrix[i, j] = data[st][bp]
            
    return matrix, st_labels, bp_labels
