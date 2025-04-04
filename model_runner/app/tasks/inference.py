from uvicorn.config import logger
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import mlflow
import os
import time
from scipy.special import softmax

# HuggingFace Model to be used for inferencing
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

class InferenceTask:

	def __init__(self):
		self.clear()
		self.load_model()

	def load_model(self):
		try:
			self.__tokenizer = AutoTokenizer.from_pretrained(MODEL)
			self.__config = AutoConfig.from_pretrained(MODEL)
			self.__model = AutoModelForSequenceClassification.from_pretrained(MODEL)
			self.__is_loaded = True
		except Exception as ex:
			logger.error("Failed to load inference model: {ex}")
			self.clear()
			return False

		return True

	def clear(self):
		self.__is_loaded = False
		self.__tokenizer = None
		self.__config = None
		self.__model = None

	def is_loaded(self):
		return self.__is_loaded

	def predict(self, messages: list[str]):
		if len(messages) == 0:
			return None

		if not self.is_loaded() and not self.load_model():
			return None

		mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
		mlflow.set_experiment("Sentiment Analysis")

		with mlflow.start_run() as run:
			preprocessed_messages = self.__preprocess(messages)
			labelized_scores = []

			for message in preprocessed_messages:
				encoded_input = self.__tokenizer(message, return_tensors='pt', padding="longest")
				output = self.__model(**encoded_input)
				scores = output[0][0].detach().numpy()
				scores = softmax(scores)
				scores = self.__labelize(scores)
				labelized_scores.append(scores)

			mean_sentiment = self.__calculate_mean_sentiment(labelized_scores)
			mean_sentiment["samples"] = len(labelized_scores)
			logger.info(mean_sentiment)

			mlflow.log_metrics(mean_sentiment, step=int(time.time()))

			return labelized_scores

	def __calculate_mean_sentiment(self, labelized_scores: list):
		total_samples = float(len(labelized_scores))

		mean_sentiment = {
			"positive": 0.0,
			"neutral": 0.0,
			"negative": 0.0,
		}

		for score in labelized_scores:
			mean_sentiment["positive"] += score["positive"]
			mean_sentiment["neutral"] += score["neutral"]
			mean_sentiment["negative"] += score["negative"]

		mean_sentiment["positive"] /= total_samples
		mean_sentiment["neutral"] /= total_samples
		mean_sentiment["negative"] /= total_samples

		return mean_sentiment

	# Preprocess text (username and link placeholders)
	def __preprocess(self, messages: list[str]):
		msg_list = []
		for message in messages:
			new_message = []
			for t in message.split(" "):
				t = '@user' if t.startswith('@') and len(t) > 1 else t
				t = 'http' if t.startswith('http') else t
				new_message.append(t)
			msg_list.append(" ".join(new_message))
		return msg_list

	def __labelize(self, scores):
		output = {}
		ranking = np.argsort(scores)
		ranking = ranking[::-1]
		for i in range(scores.shape[0]):
			l = self.__config.id2label[ranking[i]]
			s = float(scores[ranking[i]])
			output[l] = s
		return output

# Preload a global instance so that inference can be
# executed immediately when requested
infer_task = InferenceTask()
