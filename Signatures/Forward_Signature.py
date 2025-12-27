#SIG-LSTM

import matplotlib
matplotlib.use("TkAgg")  # ou "Agg" si tu ne veux pas d'interface graphique
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import iisignature
import matplotlib.pyplot as plt
import math
import torch
from torch import nn
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SIG_LSTM1(nn.Module):
    def __init__(self, in_size, hidden_size, level, device='cpu'):
        super().__init__()
        self.device = device
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.level = level
        
        # Poids pour LSTM classique
        self.Wha = nn.Parameter(torch.randn(hidden_size, hidden_size * 3) / np.sqrt(hidden_size))
        self.Wxa = nn.Parameter(torch.randn(in_size, hidden_size * 3) / np.sqrt(in_size))
        self.ba = nn.Parameter(torch.zeros(hidden_size * 3))
        
        # Calcul dimension signature
        actual_input_channels = in_size + 1  # +1 pour le temps
        self.signature_dim = iisignature.siglength(actual_input_channels, level)
        
        # Poids pour la signature
        self.Wf = nn.Parameter(torch.randn(self.signature_dim, hidden_size) / np.sqrt(hidden_size))
        self.bf = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        N, T, I = x.shape
        H = self.hidden_size
        h = [torch.zeros([N, H], device=self.device)]
        c_prev = torch.zeros([N, H], device=self.device)
        
        # Tensor temps
        time_points = torch.linspace(0, 1, T, device=self.device).unsqueeze(0).unsqueeze(2).repeat(N, 1, 1)
        full_path = torch.cat([time_points, x], dim=2)  # shape (N, T, I+1)
        
        for t in range(T):
            x_t = x[:, t, :]
            
            # Path jusqu'au temps courant
            if t == 0:
                x0t = full_path[:, 0:1, :].repeat(1, 2, 1)  # dupliquer le 1er point
            else:
                x0t = full_path[:, :t+1, :]
            
            # Signature iisignature
            sig = iisignature.sig(x0t.detach().cpu().numpy(), self.level)
            sig = torch.from_numpy(sig).float().to(self.device) / x0t.size(1)
            
            # LSTM classique
            a = torch.matmul(x_t, self.Wxa) + torch.matmul(h[t], self.Wha) + self.ba
            i_gate, o_gate, g_gate = torch.split(a, H, dim=1)
            f_gate = torch.matmul(sig, self.Wf) + self.bf
            
            i = torch.sigmoid(i_gate)
            f = torch.sigmoid(f_gate)
            o = torch.sigmoid(o_gate)
            g = torch.tanh(g_gate)
            c_next = f * c_prev + i * g
            h_next = o * torch.tanh(c_next)
            
            h.append(h_next)
            c_prev = c_next
        
        h_out = torch.stack(h[1:], dim=1)
        return h_out

#  Modèle prédiction 
class SIG_LSTM_Predictor(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, level, device='cpu'):
        super().__init__()
        self.sig_lstm = SIG_LSTM1(in_size, hidden_size, level, device=device)
        self.output_layer = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        h_out = self.sig_lstm(x)
        y_pred = self.output_layer(h_out)
        return y_pred

