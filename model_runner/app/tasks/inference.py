from uvicorn.config import logger
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import mlflow
import os
import time
from scipy.special import softmax

# HuggingFace Model to be used for inferencing
MODEL = "gpicciuca/twitter-roberta-base-sentiment-latest"
# MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class InferenceTask:

	"""
	This class encapsulates the entire inferencing logic by using HuggingFace's Transformers library.
	It offers a convenient "predict()" method that returns a list of dictionaries, where each
	dictionary contains the sentiment analysis for each message that has been evaluated.
	"""

	def __init__(self):
		self.clear()
		self.load_model()

	def load_model(self):
		"""
		Loads the classification model, its configuration and the tokenizer required for pre-processing
		any text that needs to be inferenced later on.

		Returns:
			bool: True if loading succeeded, false otherwise
		"""
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
		"""
		Resets the state of this instance
		"""
		self.__is_loaded = False
		self.__tokenizer = None
		self.__config = None
		self.__model = None

	def is_loaded(self):
		"""
		Checks if the class is ready and can be used, depending on whether 
  		a model has been loaded.

		Returns:
			bool: True if model was loaded, false otherwise
		"""
		return self.__is_loaded

	def predict(self, messages: list[str]):
		"""
		Method taking a list of messages to perform the sentiment classification on.
		Each inference run is logged in MLFlow under the experiment 'Sentiment Analysis'.
		For efficiency, only the average of the whole bulk request is logged.

		Args:
			messages (list[str]): List of messages to classify

		Returns:
			list[dict]: A list of dictionaries where each element contains the probabilities
						for 'positive', 'neutral' and 'negative' sentiment.
		"""
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
		"""
		Calculates the average sentiment over a list of classified messages.

		Args:
			labelized_scores (list): List of labelled scores resulting from the prediction step.

		Returns:
			dict: Dictionary with average values for for 'positive', 'neutral' and 'negative'.
		"""
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

	def __preprocess(self, messages: list[str]):
		"""
		Preprocesses the messages to remove certain patterns that are not
		required for inferencing. User tags and http links are stripped out.

		Args:
			messages (list[str]): List of messages to preprocess

		Returns:
			list[str]: List of processed messages without user tags and links
		"""
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
		"""
		Helper method to transform numpy labels, coming as a result of the classification,
		back into their equivalent textual version so that they are human-readable by using
		the model's configuration.

		Args:
			scores: Result from prediction for each individual message

		Returns:
			dict: Dictionary containing the sentiment prediction with human-readable labels
		"""
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
