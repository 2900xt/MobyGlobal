import requests

BASE_URL = "http://127.0.0.1:5000"

def register_sensor(name, location):
    url = f"{BASE_URL}/register"
    payload = {
        'name': name,
        'location': location
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Sensor registered successfully")
    else:
        print(f"Failed to register sensor: {response.json()}")

def update_value(name, prob):
    url = f"{BASE_URL}/update"
    payload = {
        'name': name,
        'prob': prob
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Value updated successfully")
    else:
        print(f"Failed to update value: {response.json()}")

def unregister_sensor(name):
    url = f"{BASE_URL}/unregister"
    payload = {
        'name': name
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Sensor unregistered successfully")
    else:
        print(f"Failed to unregister sensor: {response.json()}")

def get_list():
    url = f"{BASE_URL}/get_list"
    response = requests.post(url)
    if response.status_code == 200:
        print("Device list retrieved successfully")
        print(response.json())
    else:
        print(f"Failed to retrieve device list: {response.json()}")

if __name__ == "__main__":
    # Example usage:
    register_sensor("sensor1", "location1")
    update_value("sensor1", 0.75)
    get_list()
    unregister_sensor("sensor1")