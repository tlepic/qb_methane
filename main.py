import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from PIL import Image
from utils import visualize_with_data
import pandas as pd

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader

from methane import ImageDataset, weight_init
from methane.data import load_train
from methane.models import Gasnet, Gasnet2, MethaneDetectionModel, SimplifiedGasnet, TestModel
from methane.data.dataset import CheckedImageDataset

# Before starting the training loop, load existing results if they exist
if os.path.exists('results_table.csv'):
    results_df = pd.read_csv('results_table.csv')
else:
    results_df = pd.DataFrame(columns=['Image Index', 'True Label', 'Predicted Label', 'Correct'])

# Analyser les arguments de la ligne de commande
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", type=str, default="data")
ap.add_argument("--k_cv", type=int, default=5)
ap.add_argument("--batch_size", type=int, default=12)
ap.add_argument("--model", type=str, default="test")

# Configurer les journaux
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Initialisation de random seed
args = ap.parse_args()
torch.manual_seed(42)

# ======================
# Augmentation Pipeline
# ======================

def ensure_grayscale_shape(img_tensor):
    if len(img_tensor.shape) == 2:     
        return img_tensor.unsqueeze(0)  

def get_transforms(augment=True):
    """
    Returns the appropriate transforms that should be applied to the training set only.

    Args:
    - augment (bool): If True, returns augmentation pipeline, else returns basic transform.

    Returns:
    - torchvision.transforms.Compose: Transformation pipeline.
    """
    if augment:
        return transforms.Compose([
            transforms.Lambda(ensure_grayscale_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(ensure_grayscale_shape),
        ])
    
# ======================
# Model Fitting & Training
# ======================

# Étape 4 : Définir la fonction principale
def main(args):
    """
    Fonction principale pour l'entraînement et l'évaluation du modèle.

    Args:
        args (argparse.Namespace): Arguments de la ligne de commande.

    Returns:
        int: Code de retour (0 pour succès).
    """
    global results_df

    # Charger les données d'entraînement
    logging.info("Load train data")
    X_train, y_train = load_train(args.data_dir)
    # Créer le jeu de données et effectuer une validation croisée en k-fold
    logging.info("Creating dataset")

    kfold = StratifiedKFold(args.k_cv, shuffle=True, random_state=42)
    X = np.arange(len(X_train))
    acc = []
    auc = []
    
    all_misclassified_images = []
    all_titles = []
    all_data = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y_train)):
        # Data Splitting
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42, stratify=y_train[train_idx]
        )

        # Splitting the data based on indices
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        X_fold_test = X_train[test_idx]
        y_fold_test = y_train[test_idx]

        logging.info(f"Starting fold {fold+1}/{args.k_cv}")

        train_ds = CheckedImageDataset(torch.tensor(X_fold_train), torch.tensor(y_fold_train), transform=get_transforms(augment=True))
        val_ds = CheckedImageDataset(torch.tensor(X_fold_val), torch.tensor(y_fold_val), transform=get_transforms(augment=False))
        test_ds = CheckedImageDataset(torch.tensor(X_fold_test), torch.tensor(y_fold_test), transform=get_transforms(augment=False))

        # DataLoader Initialization
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        # Logging Dataset Sizes
        logging.info(f"The train_ds size {len(train_ds)}")
        logging.info(f"The val_ds size {len(val_ds)}")
        logging.info(f"The test_ds size {len(test_ds)}")
        
        # ======================
        # Model Initialization and Training
        # ======================

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            filename="best-model-{epoch:02d}-{val_loss:.2f}",
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=10, 
            mode="min", 
            verbose=True
        )
        
        trainer = pl.Trainer(
            max_epochs=100,
            callbacks=[
                early_stopping_callback, 
                checkpoint_callback
                ],
            log_every_n_steps=5
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

        # ======================
        # Evaluation
        # ======================

        predictions = []
        ground_truth = []
        probas = []
        for batch in output:
            proba, y_hat, y = batch
            predictions.extend(y_hat.cpu().numpy())
            ground_truth.extend(y.cpu().numpy())
            probas.extend(proba.cpu().numpy())

        
        for i, ((proba, y_hat, y), batch_indices) in enumerate(zip(output, test_idx)):
            correct = (y_hat.cpu().squeeze() == y.cpu().squeeze()).numpy().astype(int)
            batch_df = pd.DataFrame({
                'Image Index': batch_indices,
                'True Label': y.cpu().squeeze().numpy(),
                'Predicted Label': y_hat.cpu().squeeze().numpy(),
                'Correct': correct
            })
            results_df = pd.concat([results_df, batch_df], ignore_index=True)
    
        # Save the results table to a CSV file for later reference (optional)
        results_df.to_csv('results_table.csv', index=False)

        all_images = [X_train[idx] for idx in results_df['Image Index']]
        titles = [f"True: {row['True Label']}, Pred: {row['Predicted Label']}" for _, row in results_df.iterrows()]
        accumulated_accuracy = np.mean(results_df['Correct'].cumsum().values / (results_df.index + 1))
        data = [f"Accumulated Accuracy: {accumulated_accuracy:.2%}" for _ in results_df.iterrows()]

        # Visualize all images with accumulated data
        visualize_with_data(all_images, titles, data)

        logging.info("Classification report")
        logging.info(classification_report(ground_truth, predictions))
        logging.info(f"ROC-AUC {roc_auc_score(ground_truth, probas)}")

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

    # Accumulate accuracy data
    accumulated_accuracy = np.mean(results_df['Correct'].cumsum().values / (results_df.index + 1))

    # Assuming X_train is a list/array of images
    all_images = [X_train[idx] for idx in results_df['Image Index']]
    titles = [f"True: {row['True Label']}, Pred: {row['Predicted Label']}" for _, row in results_df.iterrows()]
    data = [f"Accumulated Accuracy: {accumulated_accuracy:.2%}" for _ in misclassified_rows.iterrows()]

    # Visualize misclassified images with accumulated data
    visualize_with_data(all_images, titles, data)

    return 0


# Exécuter la fonction principale
if __name__ == "__main__":
    main(args)