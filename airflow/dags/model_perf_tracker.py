from airflow.models.dag import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.providers.smtp.notifications.smtp import SmtpNotifier
from airflow.models import Variable
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_ENDPOINT = Variable.get("MLFLOW_ENDPOINT", default_var=None)

with DAG(
	dag_id="model_perf_tracker",
	start_date=datetime(2025, 4, 1),
	schedule_interval="@daily",
	catchup=False
) as dag:
	input_task = EmptyOperator(task_id="input_task")
	end_task = EmptyOperator(task_id="end_task")

	@task.branch(task_id="check_last_metrics")
	def check_last_metrics():
		# Set the tracking URI to point to your MLflow server
		mlflow.set_tracking_uri(MLFLOW_ENDPOINT)
		client = MlflowClient()

		training_experiment_id = "Sentiment Classifier Training"
		# Retrieve runs sorted by start time in descending order
		runs = client.search_runs(
			[training_experiment_id],
			order_by=["attributes.start_time DESC"]
		)

		if len(runs) < 2:
			print("Not enough runs to perform an accuracy drift check.")
			return "end_task"

		# Get the two most recent runs
		latest_run = runs[0]
		previous_run = runs[1]

		# Extract the 'accuracy' metric; adjust metric name if needed
		latest_accuracy = latest_run.data.metrics.get("accuracy")
		previous_accuracy = previous_run.data.metrics.get("accuracy")

		if latest_accuracy is None or previous_accuracy is None:
			print("Missing accuracy metric in one of the runs.")
			return "metrics_check_failed"

		drift = previous_accuracy - latest_accuracy
		print(f"Latest Accuracy: {latest_accuracy}")
		print(f"Previous Accuracy: {previous_accuracy}")
		print(f"Accuracy Drift: {drift}")

		drift_threshold = 0.05
		if drift > drift_threshold:
			print(f"Accuracy drift detected: {drift:.4f} is above the threshold of {drift_threshold}")
			return "metrics_check_failed"

		print("Accuracy drift is within the acceptable range.")
		return "end_task"

	@task(
     	task_id="metrics_check_failed",
      	on_failure_callback=SmtpNotifier(
			from_email=Variable.get("REPORT_EMAIL_FROM", default_var=None),
			to=Variable.get("REPORT_EMAIL_TO", default_var=None),
			subject="Model Training failed (#{{ ti.task_id }})"
		)
    )
	def metrics_check_failed():
		raise Exception("Metrics check failed. Accuracy worsened over the last runs.")

	check_last_metrics_task = check_last_metrics()
	metrics_check_failed_task = metrics_check_failed()

	## Chain tasks
	input_task >> check_last_metrics_task >> end_task
	check_last_metrics_task >> metrics_check_failed_task
	metrics_check_failed_task >> end_task
