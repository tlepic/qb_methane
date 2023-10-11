import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class TestModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Conv-Pool Structure 1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm2d(8)

        # Conv-Pool Structure 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.3)
        self.batchnorm2 = nn.BatchNorm2d(16)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # First Conv-Pool
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second Conv-Pool
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        acc = BinaryAccuracy()
        auc = BinaryAUROC()
        accuracy = acc(y_hat, y)
        aucroc = auc(y_hat, y)
        self.log("train_acc", accuracy, prog_bar=True)
        self.log("train_roc_auc", aucroc, prog_bar=True)
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
        acc = BinaryAccuracy()
        auc = BinaryAUROC()
        accuracy = acc(y_hat, y)
        aucroc = auc(y_hat, y)
        self.log("val_acc", accuracy, prog_bar=True)
        self.log("val_roc_auc", aucroc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.long()
        out = self(x)
        proba = F.sigmoid(out)
        y_hat = (proba > 0.5).int()
        return proba.view(-1), y_hat, y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return [optimizer], [scheduler]
