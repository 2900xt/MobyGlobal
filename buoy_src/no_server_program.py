import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import sounddevice as sd
import librosa
import threading
import datetime
import queue
from flask import request
import time
import subprocess

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

def log(msg):

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")


# Queue for communication between threads
data_queue = queue.Queue()
stop_flag = threading.Event()

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
            step = 3

            rng = range(0, n_time_frames - window_length + 1, step)

            for start_idx in rng:
                #display_progress_bar(start_idx, n_time_frames - window_length + 1)
                end_idx = start_idx + window_length
                window = mel_spectrogram_dB[:, start_idx:end_idx]  # Extract the window
                X = np.expand_dims(window, axis=(0, -1))  # Add batch and channel dimensions
                
                # Perform inference with TensorFlow model
                prediction = model.predict(X, verbose=0)
                prob = max(prob, prediction[0][0])
                    

            if prob > 0.5:
                log(f'Whale Detected!')

            log(f'{prob*100:.6f}% whale')
        except Exception:
            log('Processing thread error')
            time.sleep(1)

while True:
    log('Whale Audio Detector 0.2')

    model_path = '/home/taha/rsef2025/whale_detector/Moby5.h5'
    model = load_model(model_path)
    log(f"Using Moby5.h5")

    duration = 3.0  # Record duration in seconds
    sampling_rate = 44100  # Sampling rate

    process_thread1 = threading.Thread(target=process_audio_thread)
    process_thread1.start()
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

    log('Killing Threads')
    process_thread1.join()
    log('All threads stopped')
    exit()
