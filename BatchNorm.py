#%% Numpy implementation of Batch Normalization
import numpy as np
import torch

def batchnorm_foward(x, gamma, beta, eps):
    """
    x shape = (BatchSize, Dimension)
    gamma shape = (Dimension,)
    beta shape = (Dimension,)
    eps = 1e-5
    """

    BatchSize, Dimension = x.shape

    # Step 1: calculate mean
    mu = 1./BatchSize * np.sum(x, axis=0)
    print('mu shape: ', mu.shape)

    # Step 2: subtract mean vector of every trainings example
    xmu = x - mu
    print('xmu shape: ', xmu.shape)

    # Step 3: following the lower branch - calculation denominator
    sq = xmu ** 2
    print('sq shape: ', sq.shape)

    # Step 4: calculate variance
    var = 1./BatchSize * np.sum(sq, axis=0)
    print('var shape: ', var.shape) 

    # Step 5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)
    print('sqrtvar shape: ', sqrtvar.shape)

    # Step 6: invert sqrtwar
    ivar = 1./sqrtvar
    print('ivar shape: ', ivar.shape)

    # Step 7: execute normalization
    xhat = xmu * ivar
    print('xhat shape: ', xhat.shape)

    # Step 8: Nor the two transformation steps
    gammax = gamma * xhat
    print('gammax shape: ', gammax.shape)

    # Step 9
    out = gammax + beta
    print('out shape: ', out.shape)

    # Store intermediate
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    return out, cache


# test batchnorm_foward
batchsize, dimension = 5, 15
x = np.random.randn(batchsize, dimension)
gamma = np.ones(dimension)
beta = np.zeros(dimension)
eps = 1e-5
out, cache = batchnorm_foward(x, gamma, beta, eps)
print(out)

m = torch.nn.BatchNorm1d(dimension, eps=eps)
print(m(torch.tensor(x).float()))


#%% Torch implementation of Batch Normalization
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class MyBatchNorm(pl.LightningModule):
    def __init__(self, num_features, num_dims) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_dims = num_dims
        if self.num_dims == 2:
            self.shape = (1, num_features)
        elif self.num_dims == 4:
            self.shape = (1, num_features, 1, 1)
        
        self.gamma = nn.Parameter(torch.ones(self.shape))
        self.beta = nn.Parameter(torch.zeros(self.shape))

        self.moving_mean = torch.zeros(self.shape)
        self.moving_var = torch.ones(self.shape)


    def forward(self, x):
        if len(self.shape) == 2:
            mean = torch.mean(x, dim=0, keepdim=True)
            var = torch.mean((x - mean) ** 2, dim=0, keepdim=True)
        elif len(self.shape) == 4:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, dim=(0, 2, 3), keepdim=True)
        print(mean.shape)
        x_hat = (x - mean) / torch.sqrt(var + 1e-5)
        out = self.gamma * x_hat + self.beta
        self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
        self.moving_var = 0.9 * self.moving_var + 0.1 * var

        return out
    
    def predict_step(self, x):
        x_hat = (x - self.moving_mean) / torch.sqrt(self.moving_var + 1e-5)
        out = self.gamma * x_hat + self.beta
        return out

x = np.arange(24).reshape(2, 3, 2, 2)
print(x.shape) # BatchSize, Channel, Height, Width
mbn = MyBatchNorm(3, 4)   
print(mbn(torch.tensor(x).float()).shape)

# %%
