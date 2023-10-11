import argparse
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from methane import ImageDataset, weight_init
from methane.data import load_train
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

import cv2
import random
from src.methane.data_augmentation import rotate_image, translate_image


# Étape 1 : Analyser les arguments de la ligne de commande
ap = argparse.ArgumentParser()

ap.add_argument("--data_dir", type=str, default="data")
ap.add_argument("--k_cv", type=int, default=5)
ap.add_argument("--batch_size", type=int, default=12)
ap.add_argument("--model", type=str, default="test")

# Étape 2 : Configurer les journaux
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the log message format
)

# Étape 3 : Initialisation de random seed
args = ap.parse_args()
torch.manual_seed(42)


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
    X_train, y_train = load_train(args.data_dir)

    # Créer le jeu de données et effectuer une validation croisée en k-fold
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

        # Appliquer la data augmentation aux images
        augmented_X_fold_train = []
        augmented_y_fold_train = []
        possible_angles = [45, 90, 135]
        for image, label in zip(X_fold_train, y_fold_train):
            # Sélectionner un angle de rotation aléatoire parmi les possibilités
            random_rotation_angle = random.choice(possible_angles)
            
            # Appliquer la rotation avec l'angle aléatoire
            augmented_image = rotate_image(image, angle=random_rotation_angle)
            
            augmented_X_fold_train.append(augmented_image)
            augmented_y_fold_train.append(label)
    
        # Convertir les listes en tableaux NumPy
        augmented_X_fold_train = np.array(augmented_X_fold_train)
        augmented_y_fold_train = np.array(augmented_y_fold_train)

        # Fusionner les données d'entraînement d'origine avec les données augmentées
        X_fold_train = np.concatenate((X_fold_train, augmented_X_fold_train), axis=0)
        y_fold_train = np.concatenate((y_fold_train, augmented_y_fold_train), axis=0)

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
            max_epochs=100,  # Theo had 1
            callbacks=[early_stopping_callback, checkpoint_callback],
            log_every_n_steps=5,
        )

        if args.model == "baseline":
            model = MethaneDetectionModel()
        elif args.model == "gasnet":
            model = Gasnet()
        elif args.model == "simple-gasnet":
            model = SimplifiedGasnet()
        elif args.model == "gasnet_2":
            model = Gasnet2()
        elif args.model == "test":
            model = TestModel()
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

    return 0


# Exécuter la fonction principale
if __name__ == "__main__":
    logging.info("Create dataset")
    main(args)
