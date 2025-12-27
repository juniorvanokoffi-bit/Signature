import torch
import torch.nn as nn
from Forward_Signature import SIG_LSTM_Predictor
from Simulation import generate_brownian_motion
from visualisation import plot_prediction

#  Paramètres user
in_size = 1
hidden_size = 16
out_size = 1
level = 2
device = 'cpu'
epochs = 200
learning_rate = 0.01

#Génération des données 
X, y_true = generate_brownian_motion(T=50, seed=1)

#  Création du modèle 
model = SIG_LSTM_Predictor(in_size, hidden_size, out_size, level, device=device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#  Entraînement 
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Prédiction et visualisation -
y_pred = model(X).detach()
plot_prediction(y_true, y_pred)

