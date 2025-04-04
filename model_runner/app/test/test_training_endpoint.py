from .fixture import app_client
import json
import time

def test_training_endpoint(app_client):
	"""
	Checks whether the training endpoint correctly receives, starts
	and clears a training task instance.

	Given:
	- Launched training instance

	When:
	- State polled multiple times

	Then:
	- Expect state returned on first poll and instance gone on second poll
	"""
	response = app_client.post("/train/start")
	assert response.status_code == 200
	output : dict = json.loads(response.content)
	assert len(output.keys()) == 1
	assert output["message"] == "Model training was scheduled and will begin shortly."

	time.sleep(5)

	response = app_client.post("/train/get_state")
	assert response.status_code == 200
	output : dict = json.loads(response.content)
	assert len(output.keys()) == 2
	assert output["done"]
	assert not output["error"]

	time.sleep(1)

	response = app_client.post("/train/get_state")
	assert response.status_code == 200
	output : dict = json.loads(response.content)
	assert len(output.keys()) == 1
	assert output["message"] == "No training instance running!"
