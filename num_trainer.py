import os
import joblib
import pandas as pd
import logging
import yaml
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from methane.data.num_dataselect import NumDataSelector

# Set up argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("--data_file", type=str, default="./data/train_data/metadata_augmented.csv", help="Path to the data file")
ap.add_argument("--config", type=str, default="./num_models/config/config.yaml", help="Path to the configuration file")
ap.add_argument("--save_dir", type=str, default="./num_models/trained", help="Directory to save trained models")
args = ap.parse_args()

# Ensure save directory exists
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Load hyperparameters from configuration file
with open(args.config, "r") as config_file:
    config = yaml.safe_load(config_file)
rf_params = config["random_forest"]

# Load data
logging.info("Loading data...")
df = pd.read_csv(args.data_file)
logging.info("Parsing data...")
selector = NumDataSelector(df)
X, y = selector.parse_dataset()

# Implement k-fold cross-validation
kfold = StratifiedKFold(n_splits=config["k_fold"]["n_splits"], shuffle=True)
accuracies = []
f1_scores = []
aucs = []
oob_scores = []  # Out-of-bag scores

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    X_train, X_val, y_train, y_val = train_test_split(X.iloc[train_idx], y.iloc[train_idx], test_size=config["train_test_split"]["test_size"])

    # Train Random Forest model
    logging.info(f"Training model for fold {fold + 1}...")
    model = RandomForestClassifier(**rf_params, oob_score=True)
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(args.save_dir, f"random_forest_fold_{fold + 1}.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Model for fold {fold + 1} saved at {model_path}")

    # Evaluate model
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    # Log metrics for current fold
    logging.info(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}")
    logging.info(f"Fold {fold + 1} - F1 Score: {f1:.4f}")
    logging.info(f"Fold {fold + 1} - AUC: {auc:.4f}")

    accuracies.append(accuracy)
    f1_scores.append(f1)
    aucs.append(auc)
    oob_scores.append(model.oob_score_)

# Display aggregated results
logging.info(f"Average Accuracy: {sum(accuracies) / len(accuracies):.4f}")
logging.info(f"Average F1 Score: {sum(f1_scores) / len(f1_scores):.4f}")
logging.info(f"Average AUC: {sum(aucs) / len(aucs):.4f}")
logging.info(f"Average OOB Score: {sum(oob_scores) / len(oob_scores):.4f}")
