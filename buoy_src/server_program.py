import librosa
import threading
import datetime
import queue
import time
import requests
import json
import numpy as np
import sounddevice as sd
from flask import Flask, jsonify, request

# Queue for communication between threads
output_queue = queue.Queue()

# The URL of the Flask server's update endpoint
url = 'http://127.0.0.1:5000/'
dev_name = 'taha-RPi'
location = '39.042388, -77.550108'

duration = 3.0  # Record duration in seconds
sampling_rate = 44100  # Sampling rate

def log(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")


def get_microphone_audio(duration, sampling_rate):
    # Record audio from the microphone
    audio = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to complete

    # Flatten the audio array
    audio = audio.flatten()

    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sampling_rate,
        fmax=2048
    )
    mel_spectrogram_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return audio


def get_melspectrogram(audio):
    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sampling_rate,
        fmax=2048
    )
    mel_spectrogram_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_dB


# The value you want to send
def send_value(value_to_send):

    # Create a JSON payload
    payload = {'name' : dev_name, 'data' : value_to_send}

    # Send a POST request to the update endpoint
    response = requests.post(url + '/update', json=payload)

    # Print the response from the server
    if response.status_code != 200:
        log('Failed to update value' + response.text)


def register_dev():
    # Create a JSON payload
    payload = {'name' : dev_name, 'location' : location}

    # Send a POST request to the update endpoint
    response = requests.post(url + '/register', json=payload)

    # Print the response from the server
    if response.status_code != 200:
        log('Failed to register: ' + response.text)

    return


def unregister_dev():
    # Create a JSON payload
    payload = {'name' : dev_name}

    # Send a POST request to the update endpoint
    response = requests.post(url + '/unregister', json=payload)

    # Print the response from the server
    if response.status_code != 200:
        log('Failed to unregister' + response.text)

    return


def send_to_server_thread():
    while True:
        if output_queue.empty():
            time.sleep(0.1)
            continue

        audio_data = output_queue.get()
        get_melspectrogram(audio_data).tolist()
        log('Sent data to server')


log('Whale Audio Detector Client 1.0')

#register_dev()
send_thread1 = threading.Thread(target=send_to_server_thread)
send_thread1.start()

while True:
    audio_data = get_microphone_audio(duration, sampling_rate)
    output_queue.put(audio_data, timeout=1)

unregister_dev()
log('Killing Threads')
process_thread1.join()
log('All threads stopped')
exit()
