import json
import mlflow
import logging
from mlflow.tracking import MlflowClient

# ===============================
# MLflow Tracking URI
# ===============================
mlflow.set_tracking_uri("http://54.91.250.234:5000/")

# ===============================
# Logging configuration
# ===============================
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load model info (run_id, model_path, model_name) from JSON file."""
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    logger.debug('Model info loaded from %s', file_path)
    return model_info


def register_model(model_name: str, model_info: dict):
    """Register the model and assign an alias (staging)."""
    client = MlflowClient()
    run_id = model_info["run_id"]
    model_path = model_info.get("model_path", "model")

    model_uri = f"runs:/{run_id}/{model_path}"
    logger.debug(f"Registering model from URI: {model_uri}")

    # Register the model
    model_version = mlflow.register_model(model_uri, model_name)
    logger.info(f"Created version '{model_version.version}' of model '{model_name}'")

    # Assign alias instead of deprecated stage
    alias = "staging"  #  Assign alias (default = staging, but can also be 'production')
    client.set_registered_model_alias(model_name, alias, model_version.version)
    logger.debug(f"Alias '{alias}' now points to version {model_version.version} of '{model_name}'")


def main():
    try:
        model_info = load_model_info("reports/experiment_info.json")
        model_name = model_info.get("model_name", "my_model")
        register_model(model_name, model_info)
    except Exception as e:
        logger.error("Failed to complete the model registration process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
