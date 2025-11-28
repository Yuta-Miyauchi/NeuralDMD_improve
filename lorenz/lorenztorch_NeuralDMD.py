####################################################################################################
## Lorenz Prediction by Neural DMD
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import scipy as sp 
import torch 

##################################################
## Neural DMD Architecture
##################################################

def low_rank_approximation(s, low_rank = 0.9):
    ratio_s = s/s.sum()
    cumulative_s = torch.cumsum(ratio_s, dim = 0)
    idx = torch.nonzero(cumulative_s >= low_rank, as_tuple = False)
    if len(idx) == 0:
        return len(s)
    else:
        return idx[0].item() + 1

def DMD_LongTermPrediction(idx, X):
    X1 = X[:, :(X.shape[1]//2)]
    X2 = X[:, (X.shape[1]//2):]

    U, s, Vh = torch.linalg.svd(X1, full_matrices = False)
    V = Vh.T 

    r = low_rank_approximation(s)
    U = U[:, :r]
    s = s[:r]
    V = V[:, :r]

    s_inv = 1/s
    S_inv = torch.diag(s_inv)

    Atilde = U.T@X2@V@S_inv

    idx0 = idx[0]
    z0 = U.T@X1[:, 0].view([-1, 1])
    Z_pred = z0

    for i in range(1, idx.shape[0]):
        Z_pred = torch.cat([Z_pred, torch.matrix_power(Atilde, int(idx[i] - idx0))@z0], dim = 1)
    X_pred = U@Z_pred

    return X_pred

class NeuralDMD(torch.nn.Module):
    def __init__(self, original_dim, latent_dim):
        super(NeuralDMD, self).__init__()

        self.original_dim = original_dim
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.original_dim, self.latent_dim),
            torch.nn.ReLU(0.01),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),

            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),

            torch.nn.Linear(self.latent_dim, self.original_dim)
        )

    def forward(self, X):
        idx, X = X 

        Z = self.encoder(X)
        
        Z_transpose = Z.T 
        Z_pred_transpose = DMD_LongTermPrediction(idx, Z_transpose)
        Z_pred = Z_pred_transpose.T 

        X_pred = self.decoder(Z_pred)

        return X_pred

##################################################
## Data Preprocessing
##################################################

def lorenz(x, y, z, sigma = 10.0, rho = 28.0, beta = 8/3):
    dx = sigma*(y - x)
    dy = x*(rho - z) - y
    dz = x*y - beta*z
    return dx, dy, dz

def simulate_lorenz(x0 = 1.0, y0 = 1.0, z0 = 1.0, dt = 0.01, steps = 4000):
    data = np.zeros((3, steps + 1))

    data[0, 0], data[1, 0], data[2, 0] = x0, y0, z0
    for i in range(steps):
        dx, dy, dz = lorenz(data[0, i], data[1, i], data[2, i])
        data[0, i + 1] = data[0, i] + dx*dt
        data[1, i + 1] = data[1, i] + dy*dt
        data[2, i + 1] = data[2, i] + dz*dt

    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = "3d")
    ax.plot(data[0], data[1], data[2], linewidth = 0.5)

    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig('lorenz.pdf', format = 'pdf')
    plt.close()

    return data

def create_train(data):
    T = data.shape[1]
    idx = np.arange(1, int(0.7*T))

    np.random.seed(None)
    sampled = np.random.choice(idx, size = idx.shape[0], replace = False)
    sampled = np.concatenate(([0], sampled))
    sampled_next = sampled + 1
    sampled_all = np.concatenate([sampled, sampled_next], axis = 0)

    X1 = data[:, sampled]
    X2 = data[:, sampled_next]
    X = np.transpose(np.concatenate([X1, X2], axis = 1))

    return [sampled_all, X]

def create_validate(data):
    T = data.shape[1]
    idx = np.arange(int(0.7*T), int(0.8*T))
    idx1 = idx[:-1]
    idx2 = idx[1:]
    idx_all = np.concatenate([idx1, idx2], axis = 0)

    X1 = data[:, idx1]
    X2 = data[:, idx2]
    X = np.transpose(np.concatenate([X1, X2], axis = 1))

    return [idx_all, X]

def create_test(data):
    T = data.shape[1]
    idx = np.arange(int(0.8*T), T)
    idx1 = idx[:-1]
    idx2 = idx[1:]
    idx_all = np.concatenate([idx1, idx2], axis = 0)

    X1 = data[:, idx1]
    X2 = data[:, idx2]
    X = np.transpose(np.concatenate([X1, X2], axis = 1))

    return [idx_all, X]

##################################################
## For Training
##################################################

def mse_for_train(y_pred, y_true):
    err = torch.norm(y_pred - y_true, dim = 1) 
    mse = err.mean()

    return mse

def mse(y_pred, y_true):
    err = y_pred - y_true
    mse = err.pow(2).mean()

    return mse

def training(model, data, epochs = 400):
    best_loss = float('inf')
    best_state = {}
    for epoch in range(epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        model.train()
        idx, X = create_train(data)
        idx = torch.tensor(idx, dtype = torch.int64)
        X = torch.tensor(X, dtype = torch.float)
        X_pred  = model([idx, X])
        train_loss = mse_for_train(X_pred, X)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        idx, X = create_validate(data)
        idx = torch.tensor(idx, dtype = torch.int64)
        X = torch.tensor(X, dtype = torch.float)
        X_pred = model([idx, X])
        valid_loss = mse(X_pred, X)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = model.state_dict()

        if epoch == 0 or (epoch + 1)%10 == 0:
            print(f'epoch {epoch + 1: 4} / train loss: {train_loss:.6f} / valid loss: {valid_loss:.6f}')

    print(f'Best loss: {best_loss}')
    
    return best_state

##################################################
## For Evaluation
##################################################

def test(model, best_state, data):
    idx, X = create_test(data)

    model.eval()
    model.load_state_dict(best_state)
    idx = torch.tensor(idx, dtype = torch.int64)
    X = torch.tensor(X, dtype = torch.float)
    with torch.no_grad():
        X_pred_ndmd = model([idx, X])
        test_loss_ndmd = mse(X_pred_ndmd, X)

    X_pred_exact = DMD_LongTermPrediction(idx, X.T).T
    test_loss_exact = mse(X_pred_exact, X)

    print(f'test loss: Neural DMD - {test_loss_ndmd:.6f}, Exact DMD - {test_loss_exact:.6f}')

    X_pred_ndmd = X_pred_ndmd.numpy()
    X_pred_exact = X_pred_exact.numpy()

    X = X.T
    X_pred_ndmd = X_pred_ndmd.T 
    X_pred_exact = X_pred_exact.T

    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = "3d")
    ax.plot(X[0, (X.shape[1]//2):], X[1, (X.shape[1]//2):], X[2, (X.shape[1]//2):], linewidth = 0.5, label = "true")
    ax.plot(X_pred_ndmd[0, (X_pred_ndmd.shape[1]//2):], X_pred_ndmd[1, (X_pred_ndmd.shape[1]//2):], X_pred_ndmd[2, (X_pred_ndmd.shape[1]//2):], linewidth = 0.5, label = "Neural DMD")
    ax.plot(X_pred_exact[0, (X_pred_exact.shape[1]//2):], X_pred_exact[1, (X_pred_exact.shape[1]//2):], X_pred_exact[2, (X_pred_exact.shape[1]//2):], linewidth = 0.5, label = "Exact DMD")

    ax.set_title("Lorenz Attractor Prediction Result")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()
    plt.savefig('lorenz-NDMD.pdf', format = 'pdf')
    plt.close()

    return test_loss_ndmd, test_loss_exact

##################################################
## Main Function
##################################################

def main():
    data = simulate_lorenz()

    print('train')
    model = NeuralDMD(original_dim = data.shape[0], latent_dim = 10)
    best_state = training(model, data)

    print('test')
    test_loss_ndmd, test_loss_exact = test(model, best_state, data)

if __name__ == "__main__":
    main()