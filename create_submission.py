import yaml
import argparse
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from methane import ImageDataset, weight_init, seed_everything, normalize_input
from methane.data import load_train, load_test
from methane.models import (
    Gasnet,
    Gasnet2,
    MethaneDetectionModel,
    SimplifiedGasnet,
    TestModel,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader


# Getting the hyperparameters from config.yaml
with open("config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

arg_k_cv = config["arguments"]["k_cv"]
arg_batch_size = config["arguments"]["batch_size"]
arg_model = config["arguments"]["model"]

split_test_size = config["train_test_split"]["test_size"]

loaders_num_workers = config["DataLoaders"]["num_workers"]

check_callback_save_top_k = config["checkpoint_callback"]["save_top_k"]
check_callback_monitor = config["checkpoint_callback"]["monitor"]
check_callback_mode = config["checkpoint_callback"]["mode"]
check_callback_filename = config["checkpoint_callback"]["filename"]

early_callback_monitor = config["early_stopping_callback"]["monitor"]
early_callback_patience = config["early_stopping_callback"]["patience"]
early_callback_mode = config["early_stopping_callback"]["mode"]
early_callback_verbose = config["early_stopping_callback"]["verbose"]

trainer_max_epochs = config["trainer"]["max_epochs"]
trainer_callbacks = config["trainer"]["callbacks"]
trainer_log_every_n_steps = config["trainer"]["log_every_n_steps"]


# Étape 1 : Analyser les arguments de la ligne de commande
ap = argparse.ArgumentParser()

ap.add_argument("--data_dir", type=str, default="data")
ap.add_argument("--k_cv", type=int, default=arg_k_cv)
ap.add_argument("--batch_size", type=int, default=arg_batch_size)
ap.add_argument("--model", type=str, default=arg_model)
ap.add_argument("--extra", type=bool, default=False)
ap.add_argument("--norm", type=bool, default=False)
ap.add_argument("--name", type=str, default="Sample submission")

# Étape 2 : Configurer les journaux
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the log message format
)

# Étape 3 : Initialisation de random seed
args = ap.parse_args()


# Étape 4 : Définir la fonction principale
def main(args):
    logging.info("Load train data")
    X, y, X_extra = load_train(args.data_dir, extra_feature=True)
    logging.info("Load test data")
    X_test, y_test, X_extra_test, file_list = load_test(
        args.data_dir, extra_feature=True, return_path=True
    )

    # Perform train val split
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y)

    X_train = X[train_idx]
    X_extra_train = X_extra[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    X_extra_val = X_extra[val_idx]
    y_val = y[val_idx]

    moy_X = np.mean(X_train.flatten())
    std_X = np.std(X_train.flatten())

    moy_extra = np.mean(X_extra_train.flatten())
    std_extra = np.std(X_extra_train.flatten())

    if args.norm:
        print("Normalize inputs")
        for X in X_train:
            X = normalize_input(X, moy_X, std_X)
        for X in X_val:
            X = normalize_input(X, moy_X, std_X)
        for X in X_test:
            X = normalize_input(X, moy_X, std_X)

        for X in X_extra_train:
            X = normalize_input(X, moy_extra, std_extra)

        for X in X_extra_val:
            X = normalize_input(X, moy_extra, std_extra)

        for X in X_extra_test:
            X = normalize_input(X, moy_extra, std_extra)

    if args.extra:
        print("using extra data")
        # Def datasets
        train_ds = ImageDataset(
            torch.tensor(X_train),
            torch.tensor(y_train),
            extra_feature=torch.tensor(X_extra_train),
        )

        val_ds = ImageDataset(
            torch.tensor(X_val),
            torch.tensor(y_val),
            extra_feature=torch.tensor(X_extra_val),
        )
        test_ds = ImageDataset(
            torch.tensor(X_test),
            torch.tensor(y_test),
            extra_feature=torch.tensor(X_extra_test),
        )
        num_channel = 2

    else:
        train_ds = ImageDataset(
            torch.tensor(X_train),
            torch.tensor(y_train),
        )

        val_ds = ImageDataset(
            torch.tensor(X_val),
            torch.tensor(y_val),
        )
        test_ds = ImageDataset(
            torch.tensor(X_test),
            torch.tensor(y_test),
        )
        num_channel = 1

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=loaders_num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=loaders_num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=loaders_num_workers,
    )

    print(f"The train_ds size {len(train_ds)}")
    print(f"The val_ds size {len(val_ds)}")
    print(f"The test_ds size {len(test_ds)}")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=check_callback_save_top_k,
        monitor=check_callback_monitor,
        mode=check_callback_mode,
        filename=check_callback_filename,
    )

    early_stopping_callback = EarlyStopping(
        monitor=early_callback_monitor,
        patience=early_callback_patience,
        mode=early_callback_mode,
        verbose=early_callback_verbose,
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stopping_callback, checkpoint_callback],
        log_every_n_steps=5,
        accelerator=accelerator,
    )

    if args.model == "baseline":
        model = MethaneDetectionModel(num_channel)
    elif args.model == "gasnet":
        model = Gasnet(num_channel)
    if args.model == "test":
        model = TestModel(num_channel)
    else:
        print("Provide valid model name")

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
    print("Creating submission file")
    data = {"path": file_list, "label": probas}
    df = pd.DataFrame(data)
    # Specify the CSV file name
    csv_filename = "test.csv"

    # Write the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

    return 0


# Exécuter la fonction principale
if __name__ == "__main__":
    logging.info("Starting treatments")
    seed_everything(0)
    main(args)
    logging.info("End of treatments")
