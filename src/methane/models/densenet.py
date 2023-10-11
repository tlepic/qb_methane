import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class CustomDenseNet(pl.LightningModule):
    def __init__(self, pretrained=True):
        super().__init__()
        self.densenet = models.densenet121(weights=pretrained)
        self.features = self.densenet.features
        self.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.classifier = nn.Linear(
            in_features=4096, out_features=1
        )  # Adjust in_features based on your DenseNet variant

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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
