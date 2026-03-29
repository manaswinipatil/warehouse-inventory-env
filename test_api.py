import requests

BASE_URL = "http://localhost:7860"


def test_api() -> None:
    print("=" * 60)
    print("WAREHOUSE INVENTORY ENVIRONMENT - API VALIDATION")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200
    print("OK root endpoint")

    resp = requests.get(f"{BASE_URL}/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tasks"]) == 3
    print("OK tasks endpoint")

    resp = requests.post(f"{BASE_URL}/reset?task_id=0&seed=42")
    assert resp.status_code == 200
    obs = resp.json()["observation"]
    assert "robot_position" in obs
    print("OK reset endpoint")

    action = {"move_direction": 1, "action_type": 0, "target_item_id": 0}
    resp = requests.post(f"{BASE_URL}/step", json=action)
    assert resp.status_code == 200
    result = resp.json()
    assert "reward" in result
    assert "done" in result
    print(f"OK step endpoint, reward: {result['reward']:.3f}")

    resp = requests.get(f"{BASE_URL}/state")
    assert resp.status_code == 200
    print("OK state endpoint")

    resp = requests.get(f"{BASE_URL}/baseline?num_episodes=1")
    assert resp.status_code == 200
    scores = resp.json()
    print(f"OK baseline endpoint, easy score: {scores['easy']:.3f}")

    resp = requests.get(f"{BASE_URL}/metrics")
    assert resp.status_code == 200
    print("OK metrics endpoint")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - READY")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
