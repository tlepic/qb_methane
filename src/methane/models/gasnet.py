import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Gasnet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Conv-Pool Structure 1
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm2d(4)

        # Conv-Pool Structure 2
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.5)
        self.batchnorm2 = nn.BatchNorm2d(8)

        # Fully Connected Layers
        self.fc1 = nn.Linear(8 * 14 * 14, 2400)
        self.fc2 = nn.Linear(2400, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self.forward(x)
        y_hat = y_hat.view(y.shape[0])
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self.forward(x)
        y_hat = y_hat.view(y.shape[0])
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y.float())
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.long()
        out = self(x)
        proba = F.sigmoid(out)
        y_hat = (proba > 0.5).int()
        return proba.view(-1), y_hat, y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
