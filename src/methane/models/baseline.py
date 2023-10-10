import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class MethaneDetectionModel(pl.LightningModule):
    """
    Baseline model inspired from the following paper
    https://paperswithcode.com/paper/190408500
    """

    def __init__(self):
        super(MethaneDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)  # Output the likeliness of plume

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
        return y_hat, y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
