import pytest
import requests

BASE_URL = "https://jsonplaceholder.typicode.com"


@pytest.fixture(scope="session")
def api():
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    yield session
    session.close()


@pytest.fixture
def new_post(api):
    """Creates a post, yields its body, cleans up after."""
    payload = {"title": "fixture post", "body": "content", "userId": 1}
    r = api.post(f"{BASE_URL}/posts", json=payload)
    assert r.status_code == 201, "setup failed: could not create post"
    created = r.json()
    yield created
    api.delete(f"{BASE_URL}/posts/{created['id']}")


@pytest.fixture(scope="session")
def session_maker():
    print("\n>>> SESSION SETUP")
    yield
    print("\n>>> SESSION TEAR DOWN")


@pytest.fixture(scope="function")
def function_maker():
    print("\n>>> function setup")
    yield
    print("\n>>> function tear down")
