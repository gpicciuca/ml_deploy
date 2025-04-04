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

import os
import mlflow
from tasks.inference import infer_task
from config import is_test_mode

"""
Documentation:
- https://huggingface.co/docs/transformers/en//training
- https://mlflow.org/docs/latest/llms/transformers/tutorials/fine-tuning/transformers-fine-tuning
"""

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET = "zeroshot/twitter-financial-news-sentiment"
HF_DEST_REPO = "financial-twitter-roberta-sentiment"

RNG_SEED = 22

class TrainingTask:

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

	@staticmethod
	def has_instance():
		return TrainingTask.TRAINING_TASK_INST_SINGLETON is not None

	@staticmethod
	def get_instance():
		if TrainingTask.TRAINING_TASK_INST_SINGLETON is None:
			TrainingTask.TRAINING_TASK_INST_SINGLETON = TrainingTask()
		
		return TrainingTask.TRAINING_TASK_INST_SINGLETON

	@staticmethod
	def clear_instance():
		del TrainingTask.TRAINING_TASK_INST_SINGLETON
		TrainingTask.TRAINING_TASK_INST_SINGLETON = None

	def has_error(self):
		return self.__has_error

	def is_done(self):
		return self.__is_done

	def __call__(self, *args, **kwds):
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
			self.__train()
			self.__evaluate()
			self.__deploy()
		except Exception as ex:
			logger.error(f"Error during training: {ex}")
			self.__has_error = True
		finally:
			self.__is_done = True

		logout()

		self.__reload_inference_model()

	def __load_datasets(self):
		# Load the dataset.
		dataset = load_dataset(DATASET)

		# Split train/test by an 8/2 ratio.
		dataset_train_test = dataset["train"].train_test_split(test_size=0.2)
		self.__train_dataset = dataset_train_test["train"]
		self.__test_dataset = dataset_train_test["test"]

		# Swap labels so that they match what the model actually expects
		# The model expects {0: positive, 1: neutral, 2: negative}
		# But the dataset uses {0: positive, 1: negative, 2: neutral}
		# So here we just flip 1<->2 to remain consistent
		def label_filter(row):
			row["label"] = { 0: 0, 1: 2, 2: 1 }[row["label"]]
			return row

		self.__train_dataset = self.__train_dataset.map(label_filter)
		self.__test_dataset = self.__test_dataset.map(label_filter)

	def __tokenize(self):
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
		# Set the mapping between int label and its meaning.
		id2label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
		label2id = {"Bearish": 0, "Neutral": 1, "Bullish": 2}

		# Acquire the model from the Hugging Face Hub, providing label and id mappings so that both we and the model can 'speak' the same language.
		self.__model = AutoModelForSequenceClassification.from_pretrained(
			MODEL,
			num_labels=3,
			label2id=label2id,
			id2label=id2label,
		)

	def __train(self):
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
			num_train_epochs=3,
		)

		# Instantiate a `Trainer` instance that will be used to initiate a training run.
		self.__trainer = Trainer(
			model=self.__model,
			args=training_args,
			train_dataset=self.__train_tokenized,
			eval_dataset=self.__test_tokenized,
			compute_metrics=compute_metrics,
		)

		mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
		mlflow.set_experiment("Sentiment Classifier Training")

		with mlflow.start_run() as run:
			self.__run_id = run.info.run_id
			self.__trainer.train()

	def __evaluate(self):
		tuned_pipeline = pipeline(
			task="text-classification",
			model=self.__trainer.model,
			batch_size=8,
			tokenizer=self.__tokenizer,
			device="cpu", # or cuda
		)

		quick_check = (
			"I have a question regarding the project development timeline and allocated resources; "
			"specifically, how certain are you that John and Ringo can work together on writing this next song? "
			"Do we need to get Paul involved here, or do you truly believe, as you said, 'nah, they got this'?"
		)

		result = tuned_pipeline(quick_check)
		logger.debug("Test evaluation of fine-tuned model: %s %.6f" % (result[0]["label"], result[0]["score"]))

		# Define a set of parameters that we would like to be able to flexibly override at inference time, along with their default values
		model_config = {"batch_size": 8}

		# Infer the model signature, including a representative input, the expected output, and the parameters that we would like to be able to override at inference time.
		signature = mlflow.models.infer_signature(
			["This is a test!", "And this is also a test."],
			mlflow.transformers.generate_signature_output(
				tuned_pipeline, ["This is a test response!", "So is this."]
			),
			params=model_config,
		)
		
		# Log the pipeline to the existing training run
		with mlflow.start_run(run_id=self.__run_id):
			model_info = mlflow.transformers.log_model(
				transformers_model=tuned_pipeline,
				artifact_path="fine_tuned",
				signature=signature,
				input_example=["Pass in a string", "And have it mark as spam or not."],
				model_config=model_config,
			)

			# Load our saved model in the native transformers format
			loaded = mlflow.transformers.load_model(model_uri=model_info.model_uri)

			# Define a test example that we expect to be classified as spam
			validation_text = (
				"Want to learn how to make MILLIONS with no effort? Click HERE now! See for yourself! Guaranteed to make you instantly rich! "
				"Don't miss out you could be a winner!"
			)

			# validate the performance of our fine-tuning
			loaded(validation_text)

	def __deploy(self):
		self.__trainer.push_to_hub(HF_DEST_REPO)

	def __reload_inference_model(self):
		infer_task.load_model()
