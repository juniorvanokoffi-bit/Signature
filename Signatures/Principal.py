if __name__ == "__main__":
    import numpy as np
    import torch
    import iisignature
    from Sig_classic import SignatureClassique

    path = np.array([[0, 0], [1, 2], [2, 1]])
    T = 50
    delta_t = 1.0 / T

    # Mouvement brownien
    dW = np.random.randn(T) * np.sqrt(delta_t)
    W = np.cumsum(dW)

    # Supprimer le batch
    X = torch.tensor(W[:-1], dtype=torch.float32).unsqueeze(1)  # shape (T-1,1)

    # time_points correspond au temps de chaque pas
    time_points = torch.linspace(0, 1, T-1).unsqueeze(1)  # shape (T-1,1)

    # full_path = concat du temps et des valeurs
    full_path = torch.cat([time_points, X], dim=1)  # shape (T-1, 2)

    trunc = 2

    # Signature classique avec ta classe
    manager = SignatureClassique(X, trunc)
    sig = manager.calculer()

    # Signature classique avec iisignature
    sig1 = iisignature.sig(full_path.numpy(), trunc)

    print("Signature classique (iisignature) :")
    print(sig1)

