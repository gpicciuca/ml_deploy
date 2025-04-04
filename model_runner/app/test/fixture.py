import pytest
from fastapi.testclient import TestClient
from main import app
import os

@pytest.fixture()
def app_client():
	"""
	Barebone test fixture that initializes a FastAPI TestClient
	which can be used to test all endpoints provided by the application.

	Yields:
		TestClient: A client hosting the whole application so that it
					can be accessed and controlled programmatically.
	"""
	os.environ["TEST_MODE"] = "1" # Turns off actual model training
	
	client = TestClient(app)
	yield client
