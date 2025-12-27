import numpy as np
import signature.path_signature as ps

class EFMs:
    def __init__(self, path, trunc, t_grid, lam):
        self.path = path
        self.trunc = trunc
        self.t_grid = t_grid
        self.lam = lam
        self.signature_fm = None

    def calculer(self):
        """Calcule la signature à mémoire finie"""
        self.signature_fm = ps.path_to_fm_signature(
            self.path, self.trunc, self.t_grid, self.lam
        )
        return self.signature_fm

    def afficher(self):
        """Affiche les n premiers éléments de la signature à mémoire finie"""
        if self.signature_fm is None:
            print("Signature FM non calculée.")
        else:
            print( self.signature_fm[:] )


# -------------------------------
# Exemple d'utilisation
# ----------------------------                                 # paramètre lambda

# Créer une instance

# Calculer la signature


import numpy as np
import torch

np.random.seed(1)
T = 50              # nombre de pas
delta_t = 1.0 / T
trunc = 3           # troncature de la signature
lam = 0.1           # paramètre lambda

# -------------------------------
# Génération d'un mouvement brownien
# -------------------------------
dW = np.random.randn(T) * np.sqrt(delta_t)
W = np.cumsum(dW)

# -------------------------------
# Mise en forme PyTorch
# -------------------------------
X = torch.tensor(W, dtype=torch.float32).unsqueeze(0).unsqueeze(2)        # shape (1, T, 1)
time_points = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(2)          # shape (1, T, 1)

# Concaténer le temps et le chemin : shape (1, T, 2)
full_path = torch.cat([time_points, X], dim=2)

# -------------------------------
# Sous-chemin si nécessaire
# -------------------------------
x0t = full_path  # tout le chemin
x0t_np = x0t.squeeze(0).numpy()       # shape (T, 2) → requis pour EFMs
t_grid_np = np.linspace(0, 1, T)      # correspondance des temps

efm_instance = EFMs(x0t_np, trunc, t_grid_np, lam)
efm_instance.calculer()
E=efm_instance.signature_fm
print(E[:,-1])

