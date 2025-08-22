import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import yaml


# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# File path for storing errors
log_file = os.path.join("logs", "errors.log")

# Create logger
logger = logging.getLogger("data_preprocessing")
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


nltk.download('wordnet')
nltk.download('stopwords')


# --------------------------------------------------
# Function: Lemmatization
# Commit: "feat: Added lemmatization for text cleaning"
# --------------------------------------------------
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        return " ".join([lemmatizer.lemmatize(word) for word in words])
    except Exception as e:
        logger.error(f"Lemmatization error: {e}")
        return text  # fallback to original text


# --------------------------------------------------
# Function: Remove Stop Words
# Commit: "feat: Added stop words removal"
# --------------------------------------------------
def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        return " ".join([word for word in str(text).split() if word not in stop_words])
    except Exception as e:
        logger.error(f"Stopword removal error: {e}")
        return text


# --------------------------------------------------
# Function: Remove Numbers
# Commit: "feat: Added number removal from text"
# --------------------------------------------------
def removing_numbers(text):
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logger.error(f"Number removal error: {e}")
        return text


# --------------------------------------------------
# Function: Convert to Lower Case
# Commit: "feat: Added lowercase conversion"
# --------------------------------------------------
def lower_case(text):
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logger.error(f"Lowercase conversion error: {e}")
        return text


# --------------------------------------------------
# Function: Remove Punctuations
# Commit: "feat: Added punctuation removal"
# --------------------------------------------------
def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        return re.sub('\s+', ' ', text).strip()
    except Exception as e:
        logger.error(f"Punctuation removal error: {e}")
        return text


# --------------------------------------------------
# Function: Remove URLs
# Commit: "feat: Added URL removal"
# --------------------------------------------------
def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"URL removal error: {e}")
        return text


# --------------------------------------------------
# Function: Remove Short Sentences
# Commit: "feat: Added filter for short sentences"
# --------------------------------------------------
def remove_small_sentences(df):
    try:
        df.loc[df['content'].str.split().apply(len) < 3, 'content'] = np.nan
        return df
    except Exception as e:
        logger.error(f"Short sentence removal error: {e}")
        return df


# --------------------------------------------------
# Function: Normalize Text
# Commit: "feat: Added full text normalization pipeline"
# --------------------------------------------------
def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        logger.debug('Converted to lower case')

        df['content'] = df['content'].apply(remove_stop_words)
        logger.debug('Stop words removed')

        df['content'] = df['content'].apply(removing_numbers)
        logger.debug('Numbers removed')

        df['content'] = df['content'].apply(removing_punctuations)
        logger.debug('Punctuations removed')

        df['content'] = df['content'].apply(removing_urls)
        logger.debug('URLs removed')

        df['content'] = df['content'].apply(lemmatization)
        logger.debug('Lemmatization performed')

        df = remove_small_sentences(df)
        logger.debug('Small sentences removed')

        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


# --------------------------------------------------
# Main Execution
# Commit: "feat: Integrated text preprocessing pipeline"
# --------------------------------------------------
def main():
    try:
        
        # Load data
        train_data = pd.read_csv('data/raw/train.csv')
        test_data = pd.read_csv('data/raw/test.csv')
        logger.debug('Raw data loaded')

        # Normalize data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save processed data
        data_path = os.path.join("data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {data_path}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()