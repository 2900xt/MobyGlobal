import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import sounddevice as sd
import librosa
import threading
import datetime
import queue
from flask import Flask, jsonify, request
import time

def get_microphone_melspectrogram(duration, sampling_rate):
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
    
    return mel_spectrogram_dB, sampling_rate

def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def log(msg):

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")


# Queue for communication between threads
data_queue = queue.Queue()
output_queue = queue.Queue()
stop_flag = threading.Event()

import requests
import json

# The URL of the Flask server's update endpoint
url = 'https://whale-detector.vercel.app/'
# url = 'http://127.0.0.1:5000'
dev_name = 'taha-RPi'
location = '39.042388, -77.550108'

# The value you want to send
def send_value(value_to_send):

    # Create a JSON payload
    payload = {'name' : dev_name, 'prob' : value_to_send}

    # Send a POST request to the update endpoint
    response = requests.post(url + '/update', json=payload)

    # Print the response from the server
    if response.status_code != 200:
        log('Failed to update value')

def register_dev():
    # Create a JSON payload
    payload = {'name' : dev_name, 'location' : location}

    # Send a POST request to the update endpoint
    response = requests.post(url + '/register', json=payload)

    # Print the response from the server
    if response.status_code != 200:
        log('Failed to register')

    return

def unregister_dev():
    # Create a JSON payload
    payload = {'name' : dev_name}

    # Send a POST request to the update endpoint
    response = requests.post(url + '/unregister', json=payload)

    # Print the response from the server
    if response.status_code != 200:
        log('Failed to unregister')

    return

def send_to_server_thread():
    while True:
        if output_queue.empty():
            time.sleep(0.1)
            continue

        data = output_queue.get()
        send_value(float(data)) 

def process_audio_thread():
    while not stop_flag.is_set():
        try:
            if data_queue.empty():
                time.sleep(0.1)
                continue
            
            mel_spectrogram_dB = data_queue.get()
            n_time_frames = mel_spectrogram_dB.shape[1]
            window_length = 87
            prob = 0
            step = 10

            rng = range(0, n_time_frames - window_length + 1, step)

            for start_idx in rng:
                #display_progress_bar(start_idx, n_time_frames - window_length + 1)
                end_idx = start_idx + window_length
                window = mel_spectrogram_dB[:, start_idx:end_idx]  # Extract the window
                X = np.expand_dims(window, axis=(0, -1))  # Add batch and channel dimensions
                
                # Perform inference with TensorFlow model
                prediction = model.predict(X)
                print(prediction)
                prob = max(prob, 0)
                    

            if prob > 0.5:
                log(f'Whale Detected!')

            output_queue.put(prob)
            log(f'{prob*100:.6f}% whale')
        except Exception:
            log('Processing thread error')
            time.sleep(1)

while True:
    log('Whale Audio Detector 0.2')
    
    register_dev()

    model_path = './Moby5.h5'
    model = load_model(model_path)
    log(f"Using Moby5.h5")

    model.eval()
    duration = 3.0  # Record duration in seconds
    sampling_rate = 44100  # Sampling rate

    process_thread1 = threading.Thread(target=process_audio_thread)
    process_thread1.start()

    send_thread1 = threading.Thread(target=send_to_server_thread)
    send_thread1.start()

    old_spectogram = None
    try:
        while not stop_flag.is_set():
            mel_spectrogram_dB, sampling_rate = get_microphone_melspectrogram(duration, sampling_rate)
            if old_spectogram is not None:
                new_spectogram = mel_spectrogram_dB
                mel_spectrogram_dB = np.hstack((old_spectogram, new_spectogram))
                old_spectogram = new_spectogram
            data_queue.put(mel_spectrogram_dB, timeout=1)
    except:
        throw
        stop_flag.set()
        log('Recording thread error')

    unregister_dev()

    log('Killing Threads')
    process_thread1.join()
    log('All threads stopped')
    exit()
