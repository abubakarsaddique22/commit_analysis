# model building

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import yaml
import logging

# ----------------------------------------------------
# Logging Configuration
# Commit: "chore: Setup logging for console and file"
# ----------------------------------------------------
try:
    os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

    log_file = os.path.join("logs", "errors.log")

    logger = logging.getLogger("model_building")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logging setup completed successfully")
except Exception as e:
    print(f"Logging setup failed: {e}")
    raise


# ----------------------------------------------------
# Function: Load Parameters from YAML
# Commit: "feat: Add YAML parameter loading with error handling"
# ----------------------------------------------------
def load_params(params_path: str) -> dict:
    try:
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"File not found: {params_path}")

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug(f"Parameters successfully loaded from {params_path}")
        return params
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading parameters: {e}")
        raise


# ----------------------------------------------------
# Function: Load CSV Data
# Commit: "feat: Add CSV data loading with error handling"
# ----------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


# ----------------------------------------------------
# Function: Train Model
# Commit: "feat: Add Logistic Regression model training with error handling"
# ----------------------------------------------------
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    try:
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train, y_train)
        logger.debug("Model training completed successfully")
        return clf
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


# ----------------------------------------------------
# Function: Save Model
# Commit: "feat: Add model saving with error handling"
# ----------------------------------------------------
def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving model: {e}")
        raise


# ----------------------------------------------------
# Main Pipeline
# Commit: "feat: Add main pipeline for model building with exception handling"
# ----------------------------------------------------
def main():
    try:
        logger.info("Model building process started")

        train_data = load_data('data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)
        save_model(clf, 'models/model.pkl')

        logger.info("Model building process completed successfully")
    except Exception as e:
        logger.error(f"Model building process failed: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
