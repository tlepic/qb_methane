import argparse
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from methane import ImageDataset, weight_init
from methane.data import load_train
from methane.models import MethaneDetectionModel, Gasnet
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()

ap.add_argument("--data_dir", type=str, default="data")
ap.add_argument("--k_cv", type=int, default=5)
ap.add_argument("--batch_size", type=int, default=12)
ap.add_argument("--model", type=str, default="gasnet")

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the log message format
)

args = ap.parse_args()
torch.manual_seed(42)


def main(args):
    logging.info("Load train data")
    X_train, y_train = load_train(args.data_dir)

    logging.info("Creating dataset")
    kfold = StratifiedKFold(args.k_cv, shuffle=True, random_state=42)

    X = np.arange(len(X_train))
    acc = []
    auc = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y_train)):
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42, stratify=y_train[train_idx]
        )
        print("---------------------------\n")
        print(f"Starting fold {fold+1}/{args.k_cv}")
        # set the training and validation folds
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        X_fold_test = X_train[test_idx]
        y_fold_test = y_train[test_idx]

        # Def datasets
        train_ds = ImageDataset(torch.tensor(X_fold_train), torch.tensor(y_fold_train))
        val_ds = ImageDataset(torch.tensor(X_fold_val), torch.tensor(y_fold_val))
        test_ds = ImageDataset(torch.tensor(X_fold_test), torch.tensor(y_fold_test))

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

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            filename="best-model-{epoch:02d}-{val_loss:.2f}",
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        )

        trainer = pl.Trainer(
            max_epochs=100,
            callbacks=[early_stopping_callback, checkpoint_callback],
            log_every_n_steps=5,
        )

        if args.model == "baseline":
            model = MethaneDetectionModel()
        if args.model == "gasnet":
            model = Gasnet()
        else:
            print("Provide valid model name")
            break
        print("Initialize model")
        model.apply(weight_init)
        trainer.fit(model, train_loader, val_loader)
        output = trainer.predict(model, test_loader, ckpt_path="best")

        predictions = []
        ground_truth = []
        probas = []

        for batch in output:
            proba, y_hat, y = batch
            predictions.extend(y_hat.cpu().numpy())
            ground_truth.extend(y.cpu().numpy())
            probas.extend(proba.cpu().numpy())

        print("---------------------------\n")
        print("Classification report")
        print(classification_report(ground_truth, predictions))
        print(f"ROC-AUC {roc_auc_score(ground_truth, probas)}")
        print("---------------------------\n")

        acc.append(accuracy_score(ground_truth, predictions))
        auc.append(roc_auc_score(ground_truth, probas))

    print("---------------------------\n")
    print("Averaged results")
    print(
        "Average accuracy "
        + "{:.2%}".format(np.mean(np.array(acc)))
        + f" ± {np.std(np.array(acc))}"
    )
    print(
        "Average ROC AUC "
        + "{:.2%}".format(np.mean(np.array(auc)))
        + f" ± {np.std(np.array(auc))}"
    )
    print("---------------------------\n")

    return 0


if __name__ == "__main__":
    logging.info("Create dataset")
    main(args)
