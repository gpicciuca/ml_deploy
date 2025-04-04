from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from uvicorn.config import logger
import os
import argparse
from tasks.training import TrainingTask
from config import enable_test_mode

app = FastAPI()

@app.post("/train/start", response_class=JSONResponse)
async def start_model_training(background_tasks: BackgroundTasks):
	"""
	Endpoint on which a request can be sent to start model re-training, 
	if there's no training task currently running.
	The task will be carried out in background and its status can be
	polled via /train/get_state.

	Args:
		background_tasks (BackgroundTasks): BG Tasks scheduler provided by FastAPI

	Returns:
		dict: A dictionary containing a message of the outcome for the request.
	"""
	if not TrainingTask.has_instance():
		background_tasks.add_task(TrainingTask.get_instance())

		return {
			"message": "Model training was scheduled and will begin shortly.",
		}
	
	return {
		"message": "A training instance is already running.",
	}

@app.post("/train/get_state", response_class=JSONResponse)
async def poll_model_training_state():
	"""
	Checks if there is currently a training task ongoing.
	If so, returns whether it's done and/or if an error occurred.
	Otherwise if no instance is running, returns only a message.

	Returns:
		dict: Dictionary containing either done/error or message.
	"""
	if TrainingTask.has_instance():
		train_instance : TrainingTask = TrainingTask.get_instance()
		is_done = train_instance.is_done()
		has_error = train_instance.has_error()

		if is_done:
			TrainingTask.clear_instance()

		return {
			"done": is_done,
			"error": has_error,
		}

	return {
		"message": "No training instance running!",
	}

class InferenceRequest(BaseModel):
	"""
	Provides a model/schema for the accepted request body for incoming
	inference requests.
	"""
	messages: list[str]

@app.post("/inference", response_class=JSONResponse)
async def inference(data: InferenceRequest):
	"""
	

	Args:
		data (InferenceRequest): Structure containing a list of 
  								 messages that shall be evaluated

	Returns:
		json: A json list containing the sentiment analysis for each message.
			  Each element consists of a dictionary with the following keys:
			  positive, neutral, negative
	"""
	
	from tasks.inference import infer_task
	return infer_task.predict(data.messages)

@app.get("/", response_class=HTMLResponse)
async def root():
	"""
	The root endpoint for our hosted application. Only shows a message
	showing that it's up and running.

	Returns:
		str: A html response containing a hello world-like string
	"""
	return "Hi there! It's a nice blank page, isn't it?"

if __name__ == "__main__":
	"""
	Entrypoint for the application executed via command-line.
	It accepts an optional argument "--test" to enable the test mode.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("test", nargs="?", default="no")
	args = parser.parse_args()

	if args.test == "yes":
		enable_test_mode()

	config = uvicorn.Config("main:app", host="0.0.0.0", port=int(os.environ["APP_LISTEN_PORT"]), log_level="debug")
	server = uvicorn.Server(config)
	server.run()
