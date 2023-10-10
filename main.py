import argparse
import logging
import os
from PIL import Image
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from methane import ImageDataset, weight_init
from methane.data import load_train
from methane.models import MethaneDetectionModel
from methane.data.dataset import CheckedImageDataset

from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

ap = argparse.ArgumentParser()

ap.add_argument("--data_dir", type=str, default="data")
ap.add_argument("--k_cv", type=int, default=5)
ap.add_argument("--batch_size", type=int, default=12)

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the log message format
)

args = ap.parse_args()
torch.manual_seed(42)

# ======================
# Utility Functions
# ======================

def ensure_grayscale_shape(img_tensor):
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0)  # Add a channel dimension
    return img_tensor


class PrintShapeTransform:
    def __call__(self, img):
        print("Shape after transformations:", img.shape)
        return img

# ======================
# Augmentation Pipeline
# ======================

augmentation_pipeline = transforms.Compose([
    transforms.Lambda(ensure_grayscale_shape),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    PrintShapeTransform()  # Add this line to your transformations
])

# ======================
# Main Function
# ======================

def main(args):
    logging.info("Load train data")
    X_train, y_train = load_train(args.data_dir)
    logging.info("Stratifying data")
    
    kfold = StratifiedKFold(args.k_cv, shuffle=True, random_state=42)
    X = np.arange(len(X_train))
    acc = []
    auc = []
    
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

        # Dataset Preparation
        basic_transform = transforms.Compose([
            transforms.Lambda(ensure_grayscale_shape)
        ])

        train_ds = CheckedImageDataset(torch.tensor(X_fold_train), torch.tensor(y_fold_train), transform=augmentation_pipeline)
        val_ds = CheckedImageDataset(torch.tensor(X_fold_val), torch.tensor(y_fold_val), transform=basic_transform)
        test_ds = CheckedImageDataset(torch.tensor(X_fold_test), torch.tensor(y_fold_test), transform=basic_transform)
        
        # DataLoader Initialization
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        # Logging Dataset Sizes
        logging.info(f"The train_ds size {len(train_ds)}")
        logging.info(f"The val_ds size {len(val_ds)}")
        logging.info(f"The test_ds size {len(test_ds)}")

        # Inspect some batches from the DataLoader
        for i, (batch, _) in enumerate(train_loader):
            logging.info(f"Batch {i} shape in train_loader: {batch.shape}")
            if i > 5:  # Let's check just the first 5 batches for brevity
                break

        for i, (batch, _) in enumerate(val_loader):
            logging.info(f"Batch {i} shape in val_loader: {batch.shape}")
            if i > 5:
                break

        for i, (batch, _) in enumerate(test_loader):
            logging.info(f"Batch {i} shape in test_loader: {batch.shape}")
            if i > 5:
                break

        # ======================
        # Model Initialization and Training
        # ======================

        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)
        trainer = pl.Trainer(max_epochs=1, callbacks=[early_stopping_callback], log_every_n_steps=5)

        model = MethaneDetectionModel()
        logging.info("Initialize model")
        model.apply(weight_init)
        trainer.fit(model, train_loader, val_loader)
        output = trainer.predict(model, test_loader)

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
        
        logging.info("Classification report")
        logging.info(classification_report(ground_truth, predictions))
        logging.info(f"ROC-AUC {roc_auc_score(ground_truth, probas)}")

        acc.append(accuracy_score(ground_truth, predictions))
        auc.append(roc_auc_score(ground_truth, probas))

    logging.info("Averaged results")
    logging.info(f"Average accuracy: {np.mean(np.array(acc))*100:.2f}% ± {np.std(np.array(acc))}")
    logging.info(f"Average ROC AUC: {np.mean(np.array(auc))*100:.2f}% ± {np.std(np.array(auc))}")

    return 0


if __name__ == "__main__":
    main(args)
