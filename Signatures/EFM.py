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

    def afficher(self, n=10):
        """Affiche les n premiers éléments de la signature à mémoire finie"""
        if self.signature_fm is None:
            print("Signature FM non calculée.")
        else:
            print( self.signature_fm[:] )


# -------------------------------
# Exemple d'utilisation
# -------------------------------

path = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0]
])

trunc =2                                 # niveau de troncature
t_grid = np.linspace(0, 1, len(path))     # grille de temps
lam = 0.1                                 # paramètre lambda

# Créer une instance
efm_instance = EFMs(path, trunc, t_grid, lam)

# Calculer la signature
efm_instance.calculer()
# Afficher les 10 premiers éléments
E=efm_instance.signature_fm
print(E[:,-1])
