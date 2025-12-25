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
            print(self.signature_fm[:n])

