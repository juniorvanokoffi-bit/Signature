from Forward_Signature import SIG_LSTM1





# ---------------- Modèle prédiction ----------------
class SIG_LSTM_Predictor(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, level, device='cpu'):
        super().__init__()
        self.sig_lstm = SIG_LSTM1(in_size, hidden_size, level, device=device)
        self.output_layer = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        h_out = self.sig_lstm(x)
        y_pred = self.output_layer(h_out)
        return y_pred
                
# ---------------- Simulation mouvement brownien ----------------
if __name__ == "__main__":
    np.random.seed(1)
    T = 50
    delta_t = 1.0 / T
    dW = np.random.randn(T) * np.sqrt(delta_t)
    W = np.cumsum(dW)
            
    X = torch.tensor(W[:-1], dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1, T-1, 1)
    y_true = torch.tensor(W[1:], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            
    # ---------------- Modèle ----------------
    model = SIG_LSTM_Predictor(in_size=1, hidden_size=16, out_size=1, level=2, device='cpu')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
    # ---------------- Entraînement ----------------
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
    # ---------------- Prédiction ----------------
    y_pred = model(X).detach().squeeze().numpy()

    # ---------------- Visualisation ----------------
    plt.figure(figsize=(8,5))   
    time = np.linspace(0, 1, T-1)
    plt.plot(time, y_true.squeeze().numpy(), label="Vraie trajectoire", color='blue')
    plt.plot(time, y_pred, label="Trajectoire prédite", color='red', linestyle='--')
    plt.xlabel("Temps")
    plt.ylabel("W_t")
    plt.title("Prédiction d'un mouvement brownien avec SIG-LSTM")
    plt.legend()
    plt.show()

