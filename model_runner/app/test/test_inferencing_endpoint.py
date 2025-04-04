from .fixture import app_client
import json

def test_inference_endpoint(app_client):
	"""
	Tests the output of the inference endpoint of the application.
	
	Given:
		- Payload with valid list of messages
	When:
		- POST Request sent to inference endpoint
	Then:
		- Expect message to be classified as positive
	"""
	response = app_client.post(
		"/inference",
		headers={
			"Content-Type": "application/json",
   		},
		json={
			"messages": [
				"BTC is going to skyrocket!",
			],
		}
	)

	assert response.status_code == 200
	output = json.loads(response.content)
	assert isinstance(output, list)
	assert len(output) == 1
	assert output[0]["positive"] > output[0]["negative"] and output[0]["positive"] > output[0]["neutral"]


def test_inference_endpoint_with_wrong_payload(app_client):
	"""
	Tests the output of the inference endpoint of the application with an
	invalid payload.
	This should yield a 422 status error as FastAPI will not be able
	to translate the payload into the InferenceRequest model.
	
	Given:
		- Payload with wrong message key
	When:
		- POST Request sent to inference endpoint
	Then:
		- Expect 422 status code
	"""
	response = app_client.post(
		"/inference",
		headers={
			"Content-Type": "application/json",
   		},
		json={
			"msgs": [
				"BTC is going to skyrocket!",
			],
		}
	)

	assert response.status_code == 422 # Unprocessable entity

def test_inference_endpoint_with_no_prompt(app_client):
	"""
	Tests the output of the inference endpoint of the application
	when a valid payload is provided but with no actual messages.
	
	Given:
		- Payload without any messages
	When:
		- POST Request sent to inference endpoint
	Then:
		- Expect no error and correct format
	"""
	response = app_client.post(
		"/inference",
		headers={
			"Content-Type": "application/json",
   		},
		json={
			"messages": [],
		}
	)

	assert response.status_code == 200
	output = json.loads(response.content)
	assert output == None
