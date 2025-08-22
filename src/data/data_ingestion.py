# data_ingestion.py
import os
import numpy as np
import pandas as pd
import yaml
import logging
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
import logging
import os

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# File path for storing errors
log_file = os.path.join("logs", "errors.log")

# Create logger
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)  # Capture all logs

# Console Handler: Show debug/info logs in console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only show info and above on console

# File Handler: Store errors in file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.ERROR)  # Only store errors and above in file

# Log Format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Example usage
logger.info("This is an info message - appears in console only")
logger.error("This is an error message - saved in file and shown in console")
# ----------------------------------------------------
# Function: Load Parameters from YAML
# Commit Message: "feat: Added YAML parameter loading with error handling"
# ----------------------------------------------------
def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML file.

    Returns:
        dict: Parameters dictionary.
    """
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
# Commit Message: "feat: Implemented CSV data loading with validation"
# ----------------------------------------------------
def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file or URL.

    Args:
        data_url (str): URL or local path to CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        if not data_url.endswith('.csv'):
            raise ValueError("Provided file must be a CSV.")

        df = pd.read_csv(data_url)

        if df.empty:
            raise ValueError("Loaded data is empty.")

        logger.debug(f"Data successfully loaded from {data_url}")
        return df

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


# ----------------------------------------------------
# Function: Preprocess Data
# Commit Message: "feat: Added preprocessing with filtering and encoding"
# ----------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data: remove unnecessary columns, filter sentiments,
    and encode target labels.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    try:
        required_columns = {'tweet_id', 'sentiment', 'content'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise KeyError(f"Missing required columns: {missing}")

        # Drop unnecessary column
        df.drop(columns=['tweet_id'], inplace=True, errors='ignore')

        # Keep only happiness & sadness
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]

        # Encode sentiment: happiness=1, sadness=0
        df['sentiment'] = df['sentiment'].map({'happiness': 1, 'sadness': 0})

        if df.empty:
            raise ValueError("No data left after preprocessing.")

        logger.debug("Data preprocessing completed successfully.")
        return df

    except KeyError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


# ----------------------------------------------------
# Function: Save Train/Test Data
# Commit Message: "feat: Added saving of train/test datasets"
# ----------------------------------------------------
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save train and test datasets as CSV files.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        data_path (str): Directory to save datasets.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_path = os.path.join(raw_data_path, "train.csv")
        test_path = os.path.join(raw_data_path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.debug(f"Train/Test data saved at {raw_data_path}")

    except Exception as e:
        logger.error(f"Error while saving data: {e}")
        raise


# ----------------------------------------------------
# Main Execution
# Commit Message: "feat: Integrated data ingestion pipeline"
# ----------------------------------------------------
def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

        # Load raw data
        df = load_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

        # Preprocess
        final_df = preprocess_data(df)

        # Split into train/test
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save the results
        save_data(train_data, test_data, data_path='./data')

        logger.debug("Data ingestion pipeline executed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
