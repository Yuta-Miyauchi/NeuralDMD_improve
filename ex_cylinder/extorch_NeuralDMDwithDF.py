####################################################################################################
## Neural Dynamic Mode Decomposition (Neural DMD) with Deep Feature Style Based Algorithm
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import scipy as sp 
import torch 

####################################################################################################
## Exact DMD Architecture
####################################################################################################

class ExactDMD():
    def __init__(self, ncoef):
        self.ncoef = ncoef

    def train(self, data):
        idx, X = data 
        
        X = X.T
        X1 = X[:, :(X.shape[1]//2)]
        X2 = X[:, (X.shape[1]//2):]
        K = X2@X1.T@torch.linalg.inv(X1@X1.T + X1.shape[0]*self.ncoef*torch.eye(X1.shape[0]))

        norm_err = torch.norm(X2- K@X1, dim = 0)
        mse = norm_err.mean()

        return mse, K

    def predict(self, data):
        x0, idx, K = data

        idx0 = idx[0]
        x0 = x0.T 
        X_pred = x0 
        for t in range(1, idx.shape[0]):
            X_pred = torch.cat([X_pred, torch.matrix_power(K, int(idx[t] - idx0))@x0], dim = 1)
        X_pred = X_pred.T

        return X_pred 

####################################################################################################
## Neural DMD Architecture with Deep Feature Style Based Algorithm
####################################################################################################

class NeuralDMD_Stage1(torch.nn.Module):
    def __init__(self, original_dim, latent_dim, ncoef):
        super(NeuralDMD_Stage1, self).__init__()

        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.ncoef = ncoef

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

    def forward(self, data):
        idx, X = data

        Z = self.encoder(X)

        Z = Z.T
        Z1 = Z[:, :(Z.shape[1]//2)]
        Z2 = Z[:, (Z.shape[1]//2):]
        K = Z2@Z1.T@torch.linalg.inv(Z1@Z1.T + Z1.shape[0]*self.ncoef*torch.eye(Z1.shape[0]))

        idx0 = idx[0]
        z0 = Z1[:, 0].view([-1, 1])
        Z_pred = z0 
        for t in range(1, idx.shape[0]):
            Z_pred = torch.cat([Z_pred, torch.matrix_power(K, int(idx[t] - idx0))@z0], dim = 1)

        norm_err = torch.norm(Z_pred - Z, dim = 0)
        mse = norm_err.mean()

        return mse, K 

class NeuralDMD_Stage2(torch.nn.Module):
    def __init__(self, original_dim, latent_dim):
        super(NeuralDMD_Stage2, self).__init__()

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

    def forward(self, data):
        x0, idx, K = data

        z0 = self.encoder(x0)

        idx0 = idx[0]
        z0 = z0.T 
        Z_pred = z0 
        for t in range(1, idx.shape[0]):
            Z_pred = torch.cat([Z_pred, torch.matrix_power(K, int(idx[t] - idx0))@z0], dim = 1)
        Z_pred = Z_pred.T 

        X_pred = self.decoder(Z_pred)

        return X_pred

####################################################################################################
## Data Preprocessing
####################################################################################################

def load_sampled_vortacity(file_name = '../../DATA/FLUIDS/CYLINDER_ALL.mat', sampling_ratio = 0.001):
    all_data = sp.io.loadmat(file_name)
    data = all_data['VORTALL']

    idx = np.arange(0, data.shape[0])
    sampled = np.random.choice(idx, size = int(sampling_ratio*idx.shape[0]), replace = False)
    data_sampled = data[sampled, :]

    return data_sampled

def create_train(data):
    T = data.shape[1]
    idx = np.arange(1, int(0.7*T))

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

####################################################################################################
## Training
####################################################################################################

def mse(y_pred, y_true):
    err = torch.norm(y_pred - y_true, dim = 1) 
    mse = err.mean()

    return mse

def training(models, data, learning_rate = 1e-3, epochs = 1000, cnt_stage1 = 5, cnt_stage2 = 5):
    model_stage1, model_stage2 = models

    best_loss = float('inf')
    best_state = {}

    for epoch in range(epochs):
        optimizer = torch.optim.Adam(model_stage1.parameters(), lr = learning_rate)
        model_stage1.train()
        for cnt in range(cnt_stage1):
            idx, X = create_train(data)
            idx = torch.tensor(idx, dtype = torch.int64)
            X = torch.tensor(X, dtype = torch.float)

            train_loss, K_ndmd = model_stage1([idx, X])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        K_ndmd = K_ndmd.detach()
        model_stage1.eval()
        model_stage2.encoder.load_state_dict(model_stage1.encoder.state_dict())
        for p in model_stage2.encoder.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(model_stage2.decoder.parameters(), lr = learning_rate)
        model_stage2.train()
        model_stage2.encoder.eval()
        for cnt in range(cnt_stage2):
            idx, X = create_train(data)
            idx = torch.tensor(idx, dtype = torch.int64)
            X = torch.tensor(X, dtype = torch.float)

            X_pred  = model_stage2([X[0, :].view([1, -1]), idx, K_ndmd])
            train_loss = mse(X_pred, X)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        idx, X = create_validate(data)
        idx = torch.tensor(idx, dtype = torch.int64)
        X = torch.tensor(X, dtype = torch.float)

        model_stage2.eval()
        X_pred  = model_stage2([X[0, :].view([1, -1]), idx, K_ndmd])
        valid_loss = mse(X_pred, X)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = model_stage2.state_dict()

        if epoch == 0 or (epoch + 1)%100 == 0:
            print(f'epoch {epoch + 1: 4} / train loss: {train_loss:.6f} / valid loss: {valid_loss:.6f}')

    print(f'Best loss: {best_loss}')
    
    return best_state, K_ndmd

####################################################################################################
## Evaluation
####################################################################################################

def test(models, K, best_state, data):
    model_dmd, model_stage2 = models
    K_dmd, K_ndmd = K

    idx, X = create_test(data)
    idx = torch.tensor(idx, dtype = torch.int64)
    X = torch.tensor(X, dtype = torch.float)

    X_pred_dmd = model_dmd.predict([X[0, :].view([1, -1]), idx, K_dmd])
    test_loss_dmd = mse(X_pred_dmd, X)

    model_stage2.eval()
    model_stage2.load_state_dict(best_state)
    X_pred_ndmd  = model_stage2([X[0, :].view([1, -1]), idx, K_ndmd])
    test_loss_ndmd = mse(X_pred_ndmd, X)

    print(f'test loss: Neural DMD - {test_loss_ndmd:.6f}, Exact DMD - {test_loss_dmd:.6f}')

    return test_loss_ndmd, test_loss_dmd

####################################################################################################
## Main Function
####################################################################################################

def main():
    data_sampled = load_sampled_vortacity(sampling_ratio = 0.0001)

    model_dmd = ExactDMD(ncoef = 0.001)
    model_stage1 = NeuralDMD_Stage1(original_dim = data_sampled.shape[0], latent_dim = 256, ncoef = 0.01)
    model_stage2 = NeuralDMD_Stage2(original_dim = data_sampled.shape[0], latent_dim = 256)

    print('train')
    idx, X = create_train(data_sampled)
    idx = torch.tensor(idx, dtype = torch.int64)
    X = torch.tensor(X, dtype = torch.float)

    mse_dmd, K_dmd = model_dmd.train([idx, X])

    best_state, K_ndmd = training([model_stage1, model_stage2], data_sampled)

    print('test')
    test_loss_ndmd, test_loss_dmd = test([model_dmd, model_stage2], [K_dmd, K_ndmd], best_state, data_sampled)

if __name__ == "__main__":
    main()