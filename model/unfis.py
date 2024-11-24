import torch
import numpy as np
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class UNFIS(nn.Module):
    def __init__(self, in_features: int, rules: int, out_features: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features

        self.device = device
        
        self.s = nn.Parameter(torch.zeros((1, in_features, rules), **factory_kwargs))

        self.mean = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))

        self.tsk_linear = nn.Linear(
            in_features=in_features, out_features=rules * out_features, bias=True, **factory_kwargs)


    def encode(self, X):

        mean = self.mean.view(1, *self.mean.shape)
        std = self.std.view(1, *self.std.shape)

        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        y = gaussmf(X, mean, std)
        
        epsilon = 1e-1      
        zeta = torch.sigmoid(self.s)
        y = (y + epsilon) / ((1 - zeta) * y + zeta + epsilon)

        y = torch.min(y, dim=1).values

        return y
    

    def forward(self, X):
        y = self.encode(X)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)

        y = self.tsk(X, y)

        return y
    
    def tsk(self, X, y):
        X = self.tsk_linear(X)

        X = X.reshape(-1, self.rules_count, self.out_features)
        y = y.reshape(-1, self.rules_count, 1)
        X = X * y

        return X.sum(dim=1)
    
if __name__ == "__main__":
    unfis = UNFIS(3, 5, 2)
    X = torch.randn(100, 3)
    print(unfis(X).shape)
