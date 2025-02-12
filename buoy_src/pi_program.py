import torch
import numpy as np
import torch.nn as nn
import sounddevice as sd
import librosa
import threading
import datetime
import queue
import board
import busio
from adafruit_ssd1306 import SSD1306_I2C
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from flask import Flask, jsonify, request
import time
import psutil
import subprocess

class ModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=20160, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        #print(x.shape)
        x = self.block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x

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



# I2C setup
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize the OLED display
oled = SSD1306_I2C(128, 64, i2c)

# Clear the display
oled.fill(0)
oled.show()

# Create a blank image for drawing
image = Image.new("1", (oled.width, oled.height))
draw = ImageDraw.Draw(image)

# Load a font
font = ImageFont.load_default()

# Initialize a buffer to store lines of text
line_height = 10  # Font height
max_lines = oled.height // line_height - 2  # Number of lines that fit on the display
buffer = []  # Store text lines here

def get_local_ip():
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # Connect to an external IP address (doesn't have to be reachable)
        s.connect(("8.8.8.8", 80))
        # Get the IP address from the socket
        local_ip = s.getsockname()[0]
    except Exception as e:
        # Handle any exceptions and return None
        print(f"An error occurred: {e}")
        local_ip = None
    finally:
        # Close the socket connection
        s.close()
    
    return local_ip

import subprocess

def get_cpu_temperature():
    try:
        result = subprocess.run(['/usr/bin/sensors'], stdout=subprocess.PIPE, text=True, check=True)
        output = result.stdout
        # Parse the output to find the CPU temperature
        for line in output.splitlines():
            if 'temp1' in line:  # Adjust this line based on the actual output of `sensors`
                temp_str = line.split(':')[1].strip().split()[0]
                return float(temp_str.replace('°C', ''))
    except:
        return -1.0
    return 0.0

def log_to_display(message):
    global buffer
    # Add the new message to the buffer
    buffer.append(message)

    # Trim buffer to fit the screen
    if len(buffer) > max_lines:
        buffer.pop(0)  # Remove the oldest line

    # Clear the display
    draw.rectangle((0, 0, oled.width, oled.height), fill=0)

    # Draw each line in the buffer
    for i, line in enumerate(buffer):
        y = (i+2) * line_height
        draw.text((0, y), line, font=font, fill=255)
    
    draw.text((0, 0), f'CPU:  {get_cpu_temperature()} °C', font=font, fill=255)

    # Update the display
    oled.image(image)
    oled.show()

def log(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")
    log_to_display(f"{msg}")


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
                with torch.inference_mode():
                    X = torch.Tensor(np.array([np.array([window])]))
                    prediction = model(X)
                    whale_prob = softmax(np.array(prediction))[0][1]
                    prob = max(prob, whale_prob)
                    

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

    model_path = '/home/taha/whale-noise-detector/modelv1-92%.pt'
    state_dict = torch.load(model_path, weights_only=True)
    model = ModelV0(input_shape=1, hidden_units=30, output_shape=2)
    model.load_state_dict(state_dict)
    log(f"Using {model_path}")

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
