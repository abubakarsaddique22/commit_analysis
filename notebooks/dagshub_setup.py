import mlflow
import dagshub
mlflow.set_tracking_uri('https://dagshub.com/abubakarsaddique3434/commit_analysis.mlflow')
dagshub.init(repo_owner='abubakarsaddique3434', repo_name='commit_analysis', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)