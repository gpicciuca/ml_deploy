from airflow.models.dag import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.providers.smtp.notifications.smtp import SmtpNotifier
from airflow.models import Variable
from datetime import datetime, timedelta
import requests
import time

__RETRY_DELAY_ON_FAIL = 60 * 5
__RETRY_DELAY_ON_OK = 60 * 1

ML_ENDPOINT = Variable.get("ML_ENDPOINT", default_var=None)

with DAG(
	dag_id="sentiment_trainer",
	start_date=datetime(2025, 4, 1),
	schedule_interval="@weekly",
	catchup=False
) as dag:
	input_task = EmptyOperator(task_id="input_task")
	end_task = EmptyOperator(task_id="end_task")

	request_model_training_task = BashOperator(
		task_id="request_model_training",
		bash_command=f"curl -X POST '{ML_ENDPOINT}/train/start'"
	)

	@task.branch(task_id="wait_for_training_done")
	def wait_for_training_done():
		retry_count = 10

		while retry_count > 0:
			try:
				response = requests.post(
					url=f"{ML_ENDPOINT}/train/get_state"
				)
			except Exception as ex:
				print(f"An exception was thrown when sending POST request: {ex}")
				retry_count -= 1
				time.sleep(__RETRY_DELAY_ON_FAIL)
				continue

			if response.status_code != 200:
				print(f"POST request returned status code {response.status_code}")
				retry_count -= 1
				time.sleep(__RETRY_DELAY_ON_FAIL)
				continue

			data : dict = response.json()
			print(f"Training Session returned following state: {data}")

			if "done" in data and "error" in data:
				if data["done"]:
					return "training_done" if not data["error"] else "training_failed"
				# else continue polling until done = True
				time.sleep(__RETRY_DELAY_ON_OK)
			else:
				# No training instance running, abort
				print("No training instance found to be running. Aborting...")
				break

		return "training_failed"

	@task(task_id="training_done")
	def training_done():
		return

	@task(
     	task_id="training_failed", 
      	on_failure_callback=SmtpNotifier(
			from_email=Variable.get("REPORT_EMAIL_FROM", default_var=None),
			to=Variable.get("REPORT_EMAIL_TO", default_var=None),
			subject="Model Training failed (#{{ ti.task_id }})"
		)
    )
	def training_failed():
		raise Exception("Model training has failed!")

	wait_for_training_done_task = wait_for_training_done()
	training_done_task = training_done()
	training_failed_task = training_failed()

	## Chain tasks
	input_task >> request_model_training_task
	request_model_training_task >> wait_for_training_done_task
	wait_for_training_done_task >> [training_done_task, training_failed_task] >> end_task
