import mlflow
from mlflow.tracking import MlflowClient

# ===============================
# MLflow Tracking URI
# ===============================
mlflow.set_tracking_uri("http://54.91.250.234:5000/")

def promote_model(model_name: str):
    client = MlflowClient()

    # Get version currently assigned to "staging"
    staging_versions = client.get_model_version_by_alias(model_name, "staging")

    if not staging_versions:
        raise ValueError(f"No model version found with alias 'staging' for {model_name}")

    staging_version = staging_versions.version
    print(f"Staging alias currently points to version: {staging_version}")

    # Update "production" alias to point to the same version
    client.set_registered_model_alias(model_name, "production", staging_version)
    print(f"Promoted model '{model_name}' version {staging_version} → alias 'production'")

if __name__ == "__main__":
    promote_model("my_model")



