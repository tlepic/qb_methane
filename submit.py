import argparse
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from methane import ImageDataset, weight_init
from methane.data import load_test, load_train
from methane.models import Gasnet2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

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
        drop_last=False,
    )

    print(f"The train_ds size {len(train_ds)}")
    print(f"The val_ds size {len(val_ds)}")
    print(f"The test_ds size {len(test_ds)}")

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stopping_callback, checkpoint_callback],
        log_every_n_steps=5,
    )

    model = Gasnet2()
    print("Initialize model")
    model.apply(weight_init)
    trainer.fit(model, train_loader, val_loader)
    output = trainer.predict(model, test_loader, ckpt_path="best")

    predictions = []
    probas = []

    for batch in output:
        proba, y_hat, _ = batch
        predictions.extend(y_hat.cpu().numpy())
        probas.extend(proba.cpu().numpy())

    # Create a DataFrame
    data = {"path": file_list, "label": probas}
    df = pd.DataFrame(data)
    # Specify the CSV file name
    csv_filename = "test.csv"

    # Write the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

    return 0


if __name__ == "__main__":
    logging.info("Create dataset")
    main(args)
