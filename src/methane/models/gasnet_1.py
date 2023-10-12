import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SimplifiedGasnet(pl.LightningModule):
    def __init__(self, num_channel=1):
        """
        Initializes the MethaneDetectionModel.

        Args:
            num_channel (int, optional): Number of input channels. Default is 1.
        """
        super().__init__()

        # Single Conv-Pool Structure
        self.conv1 = nn.Conv2d(num_channel, 4, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)
        self.batchnorm1 = nn.BatchNorm2d(4)

        # Single Fully Connected Layer
        self.fc1 = nn.Linear(4 * 32 * 32, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)

        return x

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch (tuple): Batch containing input features and targets.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
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
        """
        Validation step of the model.

        Args:
            batch (tuple): Batch containing input features and targets.
            batch_idx (int): Index of the current batch.
        """
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self.forward(x)
        y_hat = y_hat.view(y.shape[0])
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y.float())
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        """
        Prediction step of the model.

        Args:
            batch (tuple): Batch containing input features and targets.
            batch_idx (int): Index of the current batch.

        Returns:
            tuple: Tuple containing the predicted probabilities, predicted labels, and true labels.
        """
        x, y = batch
        x = x.float()
        y = y.long()
        out = self(x)
        proba = F.sigmoid(out)
        y_hat = (proba > 0.5).int()
        return proba.view(-1), y_hat, y

    def configure_optimizers(self):
        """
    Configures the optimizer and learning rate scheduler for the model.

    Returns:
        tuple: Tuple containing the optimizer and the learning rate scheduler.
    """
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return [optimizer], [scheduler]
