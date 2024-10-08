import argparse
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
from methane import ImageDataset, seed_everything, weight_init
from methane.data import load_train
from methane.models import (
    Gasnet,
    MethaneDetectionModel,
    TestModel,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

os.chdir("..")
# Étape 1 : Analyser les arguments de la ligne de commande
ap = argparse.ArgumentParser()

ap.add_argument("--data_dir", type=str, default="data")
ap.add_argument("--k_cv", type=int, default=5)
ap.add_argument("--batch_size", type=int, default=12)
ap.add_argument("--model", type=str, default="test")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--extra", type=bool, default=False)

# Étape 2 : Configurer les journaux
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the log message format
)

# Étape 3 : Initialisation de random seed
args = ap.parse_args()


# Étape 4 : Définir la fonction principale
def main(args):
    """
    Fonction principale pour l'entraînement et l'évaluation du modèle.

    Args:
        args (argparse.Namespace): Arguments de la ligne de commande.

    Returns:
        int: Code de retour (0 pour succès).
    """

    # Charger les données d'entraînement
    logging.info("Load train data")
    X_train, y_train, X_extra_feature = load_train(args.data_dir, extra_feature=True)

    # Créer le jeu de données et effectuer une validation croisée en k-fold
    logging.info("Creating dataset")
    kfold = StratifiedKFold(args.k_cv, shuffle=True)

    X = np.arange(len(X_train))
    acc = []
    auc = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y_train)):
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, stratify=y_train[train_idx]
        )
        print("---------------------------\n")
        print(f"Starting fold {fold+1}/{args.k_cv}")
        # set the training and validation folds
        X_fold_train = X_train[train_idx]
        X_fold_extra_train = X_extra_feature[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        X_fold_extra_val = X_extra_feature[val_idx]
        y_fold_val = y_train[val_idx]
        X_fold_test = X_train[test_idx]
        X_fold_extra_test = X_extra_feature[test_idx]
        y_fold_test = y_train[test_idx]

        if args.extra:
            # Def datasets
            train_ds = ImageDataset(
                torch.tensor(X_fold_train),
                torch.tensor(y_fold_train),
                extra_feature=torch.tensor(X_fold_extra_train),
            )

            val_ds = ImageDataset(
                torch.tensor(X_fold_val),
                torch.tensor(y_fold_val),
                extra_feature=torch.tensor(X_fold_extra_val),
            )
            test_ds = ImageDataset(
                torch.tensor(X_fold_test),
                torch.tensor(y_fold_test),
                extra_feature=torch.tensor(X_fold_extra_test),
            )
            num_channel = 2

        else:
            num_channel = 1
            train_ds = ImageDataset(
                torch.tensor(X_fold_train),
                torch.tensor(y_fold_train),
            )

            val_ds = ImageDataset(
                torch.tensor(X_fold_val),
                torch.tensor(y_fold_val),
            )
            test_ds = ImageDataset(
                torch.tensor(X_fold_test),
                torch.tensor(y_fold_test),
            )

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

        roc_auc = roc_auc_score(ground_truth, probas)
        accuracy = accuracy_score(ground_truth, predictions)

        print("---------------------------\n")
        print(f"ROC-AUC {roc_auc}")
        print("---------------------------\n")

        acc.append(accuracy)
        auc.append(roc_auc)

        # Afficher les résultats agrégés
    print("---------------------------\n")
    print("Averaged results")
    print(
        "Average accuracy "
        + "{:.2%}".format(np.mean(np.array(acc)))
        + "± {:.2%}".format(np.std(np.array(acc)))
    )
    print(
        "Average ROC AUC "
        + "{:.2%}".format(np.mean(np.array(auc)))
        + "± {:.2%}".format(np.std(np.array(auc)))
    )
    print("---------------------------\n")

    return acc, auc


# Exécuter la fonction principale
if __name__ == "__main__":
    seeds = np.arange(args.seeds)
    macro_acc = []
    macro_roc = []
    for seed in seeds:
        print(f"Start test {seed+1}/{args.seeds}")
        seed_everything(seed)
        acc, auc = main(args)
        macro_acc.extend(acc)
        macro_roc.extend(auc)

    print("===================================\n")
    print("Averaged results")
    print(
        "Average accuracy "
        + "{:.2%}".format(np.mean(np.array(macro_acc)))
        + "± {:.2%}".format(np.std(np.array(macro_acc)))
    )
    print(
        "Average ROC AUC "
        + "{:.2%}".format(np.mean(np.array(macro_roc)))
        + "± {:.2%}".format(np.std(np.array(macro_roc)))
    )
    print("===================================\n")
