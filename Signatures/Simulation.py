import numpy as np
import torch

def generate_brownian_motion(T=50, seed=1):
    np.random.seed(seed)
    delta_t = 1.0 / T
    dW = np.random.randn(T) * np.sqrt(delta_t)
    W = np.cumsum(dW)
    X = torch.tensor(W[:-1], dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1, T-1, 1)
    y_true = torch.tensor(W[1:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    return X, y_true

