import numpy as np
import signature.path_signature as ps

# Définition du chemin
path = np.array([[0, 0], [1, 2], [2, 1]])

trunc = 2
lam = 0.5

# Grille de temps
t_grid = np.array([0, 1, 2], dtype=float)

# Signature classique
sig = ps.path_to_signature(path, trunc)

# Signature à mémoire finie : attention à l’ordre !
sig_fm = ps.path_to_fm_signature(path, trunc, t_grid, lam)

print("Signature classique :")
print(sig)

print("\nSignature à mémoire finie :")
print(sig_fm)

