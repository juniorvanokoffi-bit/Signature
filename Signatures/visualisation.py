import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(y_true, y_pred, title="Prédiction d'un mouvement brownien"):
    T_minus_1 = y_true.shape[1]
    time = np.linspace(0, 1, T_minus_1)
    plt.figure(figsize=(8,5))
    plt.plot(time, y_true.squeeze().numpy(), label="Vraie trajectoire", color='blue')
    plt.plot(time, y_pred.squeeze().numpy(), label="Trajectoire prédite", color='red', linestyle='--')
    plt.xlabel("Temps")
    plt.ylabel("W_t")
    plt.title(title)
    plt.legend()
    plt.show()

