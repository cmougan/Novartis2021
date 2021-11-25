#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error

from nnet import ReadDataset, Net
import time
from loss_functions import interval_score_loss

from pytorch_tabnet.tab_model import TabNetClassifier

tic = time.time()

# Read data
train_file = "data/train_1.csv"
trainset = ReadDataset(train_file, isTrain=True)
testset = ReadDataset(train_file, isTrain=False)


# Data loaders
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)
# Test set

X_train = torch.tensor(trainset.X.values)
y_train = torch.tensor(trainset.y)

X_test = torch.tensor(testset.X.values)
y_test = torch.tensor(testset.y)

clf = TabNetClassifier()  # TabNetRegressor()
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
preds = clf.predict(X_test)

print(preds)
kk
# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Neural Network
nnet = Net(trainset.__shape__()).to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(
    nnet.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000001
)


# Train the net
loss_per_iter = []
loss_per_batch = []


# Train the net
losses = []
auc_train = []
auc_test = []

# hyperparameteres
n_epochs = 20

for epoch in range(n_epochs):
    print(epoch)

    for i, (inputs, labels) in enumerate(trainloader):
        X = inputs.to(device)
        y = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forwarde
        outputs = nnet(X.float())

        # Compute diff

        loss = interval_score_loss(outputs, y.float())

        # Compute gradient
        loss.backward()

        # update weights
        optimizer.step()

        # Save loss to plot

        losses.append(loss.item())

        if i % 50 == 0:
            auc_train.append(loss.detach().numpy())
            pred = nnet(X_test.float())
            auc_test.append(interval_score_loss(pred, y_test.float()))

            # Figure
            plt.figure()
            plt.plot(auc_train, label="train")
            plt.plot(auc_test, label="test")
            plt.legend()
            plt.ylim([0, 3000])
            plt.savefig("output/auc_NN.png")
            plt.savefig("output/auc_NN.svg", format="svg")
            plt.close()

print("Elapsed time: ", np.abs(tic - time.time()))
print("done")
