from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pelutils import log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def error(y, y_hat):
    dev = 1 - y_hat / y
    return torch.abs(dev).mean()

class NN(nn.Module):
    def __init__(self, hidden=(200, 100), epochs=20):
        super().__init__()
        self.hidden = hidden
        self.lr = 1e-4
        self.epochs = epochs

    def forward(self, x):
        return self.model(x)

    def fit(self, X, y):
        X = torch.from_numpy(X.to_numpy(copy=True).astype(np.float32)).to(device)
        y = torch.from_numpy(y.astype(np.float32)).to(device)
        dataset = TensorDataset(X, y)
        layers = list()
        layer_sizes = [X.size(1)]
        layer_sizes += self.hidden
        layer_sizes += [1]
        for l1, l2 in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU())
        layers.pop()
        self.model = nn.Sequential(*layers)
        self.optim = optim.Adam(self.parameters(), lr=self.lr)
        self.to(device)
        for i in range(self.epochs):
            dataloader = iter(DataLoader(dataset, batch_size=500, shuffle=True))
            for j, (xb, yb) in enumerate(dataloader):
                y_hat = self(xb)
                loss = error(yb, y_hat)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            log.debug("%i, %.4f" % (i, loss.item()))

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.from_numpy(X.to_numpy(copy=True).astype(np.float32)).to(device)
            res = self(X).detach().cpu().numpy().ravel()
        self.train()
        return res
