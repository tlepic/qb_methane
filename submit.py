from methane import load_train, load_test, ImageDataset, weight_init

import logging
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

import pytorch_lightning as pl
from methane.models import MethaneDetectionModel
from pytorch_lightning.callbacks import EarlyStopping

ap = argparse.ArgumentParser()

ap.add_argument("--data_dir", type=str, default="data")
ap.add_argument("--k_cv", type=int, default=5)
ap.add_argument("--batch_size", type=int, default=12)

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the log message format
)

args = ap.parse_args()


def main(args):
    logging.info("Load train data")
    X_train, y_train = load_train(args.data_dir)
    logging.info("Load test data")
    X_test, y_test, file_list = load_test(args.data_dir, return_path=True)

    logging.info("Creating dataset")

    X = np.arange(len(X_train))
    print(len(X_train))
    print(X_train[400])
    train_idx, val_idx = train_test_split(
        X, test_size=0.2, random_state=42, stratify=y_train
    )
    print("---------------------------\n")
    print(f"Training the models")
    # set the training and validation folds
    X_train_final = X_train[train_idx]
    y_train_final = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    # Def datasets
    train_ds = ImageDataset(torch.tensor(X_train_final), torch.tensor(y_train_final))
    val_ds = ImageDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = ImageDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )

    print(f"The train_ds size {len(train_ds)}")
    print(f"The val_ds size {len(val_ds)}")
    print(f"The test_ds size {len(test_ds)}")

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",  # Monitor the validation loss
        patience=10,  # Number of epochs with no improvement before stopping
        mode="min",  # 'min' mode for loss (you can use 'max' for accuracy, etc.)
        verbose=True,  # Print messages about early stopping
    )

    trainer = pl.Trainer(max_epochs=1, callbacks=[early_stopping_callback])

    model = MethaneDetectionModel()
    print("Initialize model")
    model.apply(weight_init)
    trainer.fit(model, train_loader, val_loader)
    output = trainer.predict(model, test_loader)

    predictions = []

    for batch in output:
        y_hat, _ = batch
        predictions.extend(y_hat.cpu().numpy())

    print(predictions, file_list)

    return 0


if __name__ == "__main__":
    logging.info("Create dataset")
    main(args)
