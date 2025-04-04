import os

def is_test_mode():
	"""
	Checks whether test mode is enabled or not.

	Returns:
		bool: True if test mode is enabled, false otherwise
	"""
	return os.environ.get("TEST_MODE", None) == "1"

def enable_test_mode():
    os.environ["TEST_MODE"] = "1"
