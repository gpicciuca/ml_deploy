import evaluate
import numpy as np
from uvicorn.config import logger
from datasets import load_dataset
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	Trainer,
	TrainingArguments,
	pipeline,
)
from huggingface_hub import login, logout
from scipy.special import softmax

import os
import mlflow
from tasks.inference import infer_task
from config import is_test_mode
import time


# MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET = "zeroshot/twitter-financial-news-sentiment"
MODEL = "gpicciuca/sentiment_trainer"
HF_REPO = "gpicciuca/sentiment_trainer"

RNG_SEED = 22

class TrainingTask:

	"""
	Implements a sequence of actions to control the training phase of the model.
	The class implements a callable overload method which initializes the old model,
	loads and prepares datasets and proceeds with the training.
	Upon completion, the new model will be uploaded to the HuggingFace repo only
	if its accuracy did not drop compared to the old model.

	This class is managed via singleton so that there may only be one
	instance at any time, unless manually allocated.
	"""

	TRAINING_TASK_INST_SINGLETON = None
	
	def __init__(self):
		self.__is_done = False
		self.__has_error = False

		self.__train_dataset = None
		self.__test_dataset = None
		self.__tokenizer = None
		self.__train_tokenized = None
		self.__test_tokenized = None
		self.__model = None
		self.__trainer = None
		self.__run_id = None

		self.__old_accuracy = 0.0

	@staticmethod
	def has_instance():
		"""
		Checks if a global singleton instance is available

		Returns:
			bool: True if instance available, false otherwise
		"""
		return TrainingTask.TRAINING_TASK_INST_SINGLETON is not None

	@staticmethod
	def get_instance():
		"""
		Returns the globally allocated singleton instance.
		Instance will be allocated with this method if none was previously
		allocated yet.

		Returns:
			TrainingTask: Singleton instance
		"""
		if TrainingTask.TRAINING_TASK_INST_SINGLETON is None:
			TrainingTask.TRAINING_TASK_INST_SINGLETON = TrainingTask()
		
		return TrainingTask.TRAINING_TASK_INST_SINGLETON

	@staticmethod
	def clear_instance():
		"""
		Destroys the global instance
		"""
		del TrainingTask.TRAINING_TASK_INST_SINGLETON
		TrainingTask.TRAINING_TASK_INST_SINGLETON = None

	def has_error(self):
		"""
		Checks whether an error occurred during training.

		Returns:
			bool: True if an exception was raised, false otherwise
		"""
		return self.__has_error

	def is_done(self):
		"""
		Checks whether the training is done.

		Returns:
			bool: True if done, false if still ongoing.
		"""
		return self.__is_done

	def __call__(self, *args, **kwds):
		"""
		Callable overload for this class. Initiates the training sequence
		for the existing model by loading it, loading and preparing datasets,
		fine-tuning and comparing performance against old model over the test dataset.
		"""

		self.__has_error = False
		self.__is_done = False

		if is_test_mode():
			# Simulate a successful training run in test mode
			self.__has_error = False
			self.__is_done = True
			return

		login(token=os.environ["HF_ACCESS_TOKEN"])

		try:
			self.__load_datasets()
			self.__tokenize()
			self.__load_model()
			self.__check_old_accuracy()
			self.__train()
			self.__evaluate()
			self.__deploy()
		except Exception as ex:
			logger.error(f"Error during training: {ex}")
			self.__has_error = True
		finally:
			self.__is_done = True

		if self.has_error():
			logger.error("Training did not complete and terminated with an error")
		else:
			logger.info("Training completed")

		logout()

		self.__reload_inference_model()

	def __load_datasets(self, test_size_ratio=0.2):
		"""
		Loads and splits the dataset in train and test sets.
		"""
		assert (test_size_ratio > 0.0 and test_size_ratio < 1.0)

		dataset = load_dataset(DATASET)

		# Split train/test by 'test_size_ratio'
		dataset_train_test = dataset["train"].train_test_split(test_size=test_size_ratio)
		self.__train_dataset = dataset_train_test["train"]
		self.__test_dataset = dataset_train_test["test"]

		# Swap labels so that they match what the model actually expects
		# The model expects {0: negative, 1: neutral, 2: positive}
		# But the dataset uses {0: negative, 1: positive, 2: neutral}
		# So here we just flip 1<->2 to remain consistent
		def label_filter(row):
			row["label"] = { 0: 0, 1: 2, 2: 1 }[row["label"]]
			return row

		self.__train_dataset = self.__train_dataset.map(label_filter)
		self.__test_dataset = self.__test_dataset.map(label_filter)

	def __tokenize(self):
		"""
		Loads the tokenizer previously used in the pretrained model
		and uses it to tokenize the datasets so that the input to the
		model remains consistent with what it has seen in previous
		trainings.
		"""
		# Load the tokenizer for the model.
		self.__tokenizer = AutoTokenizer.from_pretrained(MODEL)

		def tokenize_function(examples):
			# Pad/truncate each text to 512 tokens. Enforcing the same shape
			# could make the training faster.
			return self.__tokenizer(
				examples["text"],
				padding="max_length",
				truncation=True,
				max_length=256,
			)

		# Tokenize the train and test datasets
		self.__train_tokenized = self.__train_dataset.map(tokenize_function)
		self.__train_tokenized = self.__train_tokenized.remove_columns(["text"]).shuffle(seed=RNG_SEED)

		self.__test_tokenized = self.__test_dataset.map(tokenize_function)
		self.__test_tokenized = self.__test_tokenized.remove_columns(["text"]).shuffle(seed=RNG_SEED)

	def __load_model(self):
		"""
		Loads the model from the repository
		"""
		# Set the mapping between int label and its meaning.
		id2label = {0: "negative", 1: "neutral", 2: "positive"}
		label2id = {"negative": 0, "neutral": 1, "positive": 2}

		# Acquire the model from the Hugging Face Hub, providing label and id mappings so that both we and the model can 'speak' the same language.
		self.__model = AutoModelForSequenceClassification.from_pretrained(
			MODEL,
			num_labels=3,
			label2id=label2id,
			id2label=id2label,
		)

	def __check_old_accuracy(self):
		"""
		Run a prediction with the old model on the tokenized test dataset
		to evaluate the model's accuracy.
    	"""
		trainer = Trainer(model=self.__model, tokenizer=self.__tokenizer)
		output = trainer.predict(self.__test_tokenized)

		# Get logits from the prediction output.
		logits = output.predictions
		# Convert logits to predicted class labels.
		preds = np.argmax(logits, axis=1)
		# Get the true labels.
		labels = output.label_ids

		# Compute accuracy.
		self.__old_accuracy = (preds == labels).mean()
		logger.info(f"Old model accuracy: {self.__old_accuracy:.4f}")

	def __train(self):
		"""
		Performs the training/fine-tuning of the loaded model using the
		tokenized train and test datasets.
		The training run will be logged on the MLFlow Dashboard.
		Uses the 'accuracy' metric to evaluate performance.
		"""
		# Define the target optimization metric
		metric = evaluate.load("accuracy")

		# Define a function for calculating our defined target optimization metric during training
		def compute_metrics(eval_pred):
			logits, labels = eval_pred
			predictions = np.argmax(logits, axis=-1)
			return metric.compute(predictions=predictions, references=labels)

		# Checkpoints will be output to this `training_output_dir`.
		training_output_dir = "/tmp/sentiment_trainer"
		training_args = TrainingArguments(
			output_dir=training_output_dir,
			eval_strategy="epoch",
			per_device_train_batch_size=8,
			per_device_eval_batch_size=8,
			logging_steps=8,
			num_train_epochs=10,
		)

		mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
		mlflow.set_experiment("Sentiment Classifier Training")

		with mlflow.start_run() as run:
			self.__run_id = run.info.run_id

			logger.info("Initializing trainer...")
			self.__trainer = Trainer(
				model=self.__model,
				args=training_args,
				train_dataset=self.__train_tokenized,
				eval_dataset=self.__test_tokenized,
				compute_metrics=compute_metrics,
			)
			logger.info("Trainer finished")

	def __evaluate(self):
		"""
		Evaluates the fine-tuned model's performance by comparing the new
		accuracy with the old one over the same test dataset.
  		"""
		logger.info("Evaluating new model's performance")

		with mlflow.start_run(run_id=self.__run_id):
			output = self.__trainer.predict(self.__test_tokenized)

			# Get logits from the prediction output.
			logits = output.predictions
			# Convert logits to predicted class labels.
			preds = np.argmax(logits, axis=1)
			# Get the true labels.
			labels = output.label_ids

			# Compute accuracy.
			new_accuracy = (preds == labels).mean()
			mlflow.log_metrics({
				"old_accuracy": self.__old_accuracy,
				"new_accuracy": new_accuracy
			}, step=int(time.time()))

			if self.__old_accuracy > new_accuracy:
				raise Exception(f"New trained model's accuracy dropped {self.__old_accuracy:.9f} -> {new_accuracy:.9f}")
			else:
				logger.info(f"New trained model's accuracy {self.__old_accuracy:.9f} -> {new_accuracy:.9f}")
	
	def __deploy(self):
		"""
		Uploads the fine-tuned model to HuggingFace
		"""
		logger.info("Deploying Model and Tokenizer to HuggingFace")
		self.__trainer.push_to_hub(HF_REPO)
		self.__tokenizer.push_to_hub(HF_REPO)

	def __reload_inference_model(self):
		"""
		Reloads the model used by the Inference class.
		"""
		logger.info("Reloading inference model")
		infer_task.load_model()
