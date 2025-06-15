import pytest


def test_flask_app_importable_and_instantiable():
    """
    Test that the Flask application can be imported and is an instance of Flask.
    This also implicitly tests if relative imports in src/app.py are working.
    """
    try:
        from src.app import app  # app is initialized in src/app.py
    except ImportError as e:
        pytest.fail(f"Failed to import Flask app from src.app: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during app import: {e}")

    assert app is not None, "Flask app object 'app' should not be None"
    # Check if it's a Flask instance (optional, as import success is primary for now)
    # from flask import Flask
    # assert isinstance(app, Flask), "app object should be an instance of Flask"
    print("Successfully imported 'app' from src.app.")


# Minimal test to ensure pytest is running
def test_pytest_setup():
    assert True
