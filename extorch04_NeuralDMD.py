####################################################################################################
## Neural Dynamic Mode Decomposition (Neural DMD) for long-term prediction
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import scipy as sp 
import torch 

##################################################
## Neural DMD Architecture
##################################################

def low_rank_approximation(s, low_rank = 0.999):
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
            # torch.nn.Dropout(0.1),

            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),
            # torch.nn.Dropout(0.1),

            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),

            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),
            # torch.nn.Dropout(0.1),

            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(0.01),
            # torch.nn.Dropout(0.1),

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

def load_sampled_vortacity(file_name = '../DATA/FLUIDS/CYLINDER_ALL.mat', sampling_ratio = 0.001):
    all_data = sp.io.loadmat(file_name)
    data = all_data['VORTALL']

    idx = np.arange(0, data.shape[0])
    sampled = np.random.choice(idx, size = int(sampling_ratio*idx.shape[0]), replace = False)
    data_sampled = data[sampled, :]

    return data_sampled

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

def training(model, data, epochs = 1000):
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

        if epoch == 0 or (epoch + 1)%100 == 0:
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
    X_pred_ndmd = model([idx, X])
    test_loss_ndmd = mse(X_pred_ndmd, X)

    X_pred_exact = DMD_LongTermPrediction(idx, X.T).T
    test_loss_exact = mse(X_pred_exact, X)

    print(f'test loss: Neural DMD - {test_loss_ndmd:.6f}, Exact DMD - {test_loss_exact:.6f}')

    return test_loss_ndmd, test_loss_exact

##################################################
## Main Function
##################################################

def main():
    data_sampled = load_sampled_vortacity(sampling_ratio = 0.001)

    print('train')
    model = NeuralDMD(original_dim = data_sampled.shape[0], latent_dim = 256)
    best_state = training(model, data_sampled)

    print('test')
    test_loss_ndmd, test_loss_exact = test(model, best_state, data_sampled)

if __name__ == "__main__":
    main()