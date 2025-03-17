import random
import time
import requests

def generate_event():
    # Generate a random latitude and longitude
    lat = random.uniform(-90, 90)
    long = random.uniform(-180, 180)
    return lat, long

if __name__ == "__main__":
    while True:
        event = generate_event()
        print(f"Event generated at: {event}")
        requests.post("http://127.0.0.2:5000/add_event", json={"lat": event[0], "long": event[1]})
        time.sleep(5)  # Simulate an event every 5 seconds