from airflow.models.dag import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.providers.smtp.notifications.smtp import SmtpNotifier
from airflow.models import Variable
from datetime import datetime, timedelta
import requests
import tweepy
import json

X_CONSUMER_KEY = Variable.get("X_CONSUMER_KEY", default_var=None)
X_CONSUMER_SECRET = Variable.get("X_CONSUMER_SECRET", default_var=None)
X_ACCESS_TOKEN = Variable.get("X_ACCESS_TOKEN", default_var=None)
X_ACCESS_TOKEN_SECRET = Variable.get("X_ACCESS_TOKEN_SECRET", default_var=None)

ML_ENDPOINT = Variable.get("ML_ENDPOINT", default_var=None)

with DAG(
	dag_id="sentiment_inspector",
	start_date=datetime(2025, 4, 1),
	schedule_interval="@daily",
	catchup=False
) as dag:
	input_task = EmptyOperator(task_id="input_task")
	end_task = EmptyOperator(task_id="end_task")

	@task.branch(task_id="do_sentiment_check")
	def do_sentiment_check():
		auth = tweepy.OAuth1UserHandler(X_CONSUMER_KEY, X_CONSUMER_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)
		api = tweepy.API(auth)

		tweets = api.search_tweets(q="profession.ai", lang="en", count=10)
		messages = [tweet.text for tweet in tweets]

		# Uncomment for testing
		# messages = [
		# 	"BTC is gonna skyrocket! Yeaaah",
		# 	"Damnit, how could that have happened? This is unbelievable. Such a pity...",
		# 	"Hey, did you hear what President XYZ did? He introduced customs duties. What the heck was he thinking?",
		# 	"Cool cool mate, wassap? Ya know, tomorrow we gonna nail it! Did you prepare those slides for the presentation?",
		# 	"NOOOOOOOOOOOOOOO WHY DID YOU DO THAT?? ARE YOU INSANE?",
		# ]

		if len(messages) > 0:
			response = requests.post(
				url=f"{ML_ENDPOINT}/inference",
				headers={
					"Content-Type": "application/json",
				},
				json={
					"messages": messages,
				}
			)

			if response.status_code != 200:
				print(f"HTTP Error {response.status_code} - Could not receive inference results")
				return "sentiment_check_failed"

			for (msg, pred) in zip(messages, json.loads(response.content)):
				print(pred, msg)
		else:
			print("Got 0 messages from the X API")
			return "sentiment_check_failed"

		return "end_task"

	@task(
     	task_id="sentiment_check_failed",
      	on_failure_callback=SmtpNotifier(
			from_email=Variable.get("REPORT_EMAIL_FROM", default_var=None),
			to=Variable.get("REPORT_EMAIL_TO", default_var=None),
			subject="Sentiment Analysis failed (#{{ ti.task_id }})"
		)
    )
	def sentiment_check_failed():
		raise Exception("HTTP Status response from Model Runner was not OK. Something happened!")

	do_sentiment_check_task = do_sentiment_check()
	sentiment_check_failed_task = sentiment_check_failed()

	## Chain tasks
	input_task >> do_sentiment_check_task >> [end_task, sentiment_check_failed_task]
	sentiment_check_failed_task >> end_task
	
