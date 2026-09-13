import pytest

BASE_URL = "https://jsonplaceholder.typicode.com"


def test_get_signle_post(api):
    r = api.get(f"{BASE_URL}/posts/1")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1
    assert "title" in body


def test_get_all_posts_returns_100(api):
    r = api.get(f"{BASE_URL}/posts/")
    assert r.status_code == 200
    assert len(r.json()) == 100


def test_create_post(api):
    payload = {"title": "test", "body": "content", "userId": 1}
    r = api.post(f"{BASE_URL}/posts/", json=payload)
    assert r.status_code == 201
    assert r.json()["title"] == payload["title"]


@pytest.mark.parametrize("post_id", [1, 50, 100], ids=["first", "middle", "last"])
def test_post_has_required_fields(api, post_id):
    r = api.get(f"{BASE_URL}/posts/{post_id}")
    assert r.status_code == 200
    body = r.json()
    for field in ("userId", "id", "title", "body"):
        assert field in body


def test_nonexistent_post_returns_404(api):
    r = api.get(f"{BASE_URL}/posts/99999")
    assert r.status_code == 404


def test_delete_item_success(api):
    payload = {"title": "test", "body": "content", "userId": 1}
    r = api.post(f"{BASE_URL}/posts/", json=payload)
    assert r.status_code == 201
    post_id = r.json()["id"]

    r = api.delete(f"{BASE_URL}/posts/{post_id}")
    assert r.status_code == 200
    assert api.get(f"{BASE_URL}/posts/{post_id}").status_code == 404


def test_put_new_title(api, session_maker):
    session_maker
    payload = {"title": "new_title"}
    r = api.put(f"{BASE_URL}/posts/50", json=payload)
    assert r.status_code == 200
    assert r.json()["title"] == payload["title"]


def test_created_post_has_id(new_post, function_maker):
    function_maker
    assert "id" in new_post
    assert new_post["title"] == "fixture post"
