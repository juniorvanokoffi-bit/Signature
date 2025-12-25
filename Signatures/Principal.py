import numpy as np
from Sig_classic import SignatureClassique
from EFM import EFMs

if __name__ == "__main__":
    path = np.array([[0, 0], [1, 2], [2, 1]])
    trunc = 2

    manager = SignatureClassique(path, trunc)
    sig = manager.calculer()

    print("Signature classique :")
    print(sig)
