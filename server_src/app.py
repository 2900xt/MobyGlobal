import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
import numpy as np
import json
import csv
from datetime import datetime
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import requests
import librosa

app = Flask(__name__)
values = dict()
spectrograms = dict()

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")


@app.route('/')
def index():
    return render_template('map.html')


def get_melspectrogram(audio):
    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=4000,
        fmax=2048
    )
    mel_spectrogram_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_dB


@app.route('/update', methods=['POST'])
def update_data():
    try:
        # Get the new value from the request JSON body
        #print the raw json data
        data = request.get_json()

        if 'data' not in data:
            return jsonify({'error': 'No array provided'}), 400
        
        if 'name' not in data:
            return jsonify({'error': 'No name provided'}), 400

        new_spectogram = np.array(data.get('data'))
        #turn the audio into a mel spectrogram
        new_spectogram = get_melspectrogram(new_spectogram)

        dev_name = str(data.get('name'))
        #stack with previous spectrogram
        if dev_name in spectrograms:
            spectrograms[dev_name] = np.concatenate((spectrograms[dev_name], new_spectogram), axis=1)
        else:
            spectrograms[dev_name] = new_spectogram

        #restrict the size of the spectrogram (spectrograms[data.get('name')]) to 128x84


        #expand the size of 2nd dimension of the spectrogram from (128x12 to 128x84)
        #do this by repeating the values in the 2nd dimension
        new_spectogram = spectrograms[data.get('name')]
        #print(new_spectogram.shape)

        if new_spectogram.shape[1] > 90:
            #restrict the size of the spectrogram to 128x90, 90 last columns
            new_spectogram = new_spectogram[:, -90:]
            prob = set_probabillity(new_spectogram, dev_name)
        else:
            prob = 0.0
            

        # Update the map with the new data
        update_map()

        return jsonify({'message': f'Data updated successfully. P(whale) = {prob}'}), 200
    except Exception as e:
        log(f'Error in update_data: {e}')
        return jsonify({'error': 'Invalid Query'}), 400


@app.route('/register', methods=['POST'])
def register_sensor():
    try:
        # Get the new value from the request JSON body
        data = request.get_json()
        dev_name = str(data.get('name'))
        dev_loc = str(data.get('location'))
        values[dev_name] = {
            'name' : dev_name,
            'prob' : 0.0,
            'location' : dev_loc,
            'last_upd' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        log(f'Register Data: {data}')

        if dev_name == "None":
            return jsonify({'error': 'Invalid Name'}), 400


        if dev_loc == "None":
            return jsonify({'error': 'Invalid Location'}), 400


        return jsonify({'message': 'Sensor Registered Successfully'}), 200
    except Exception:
        return jsonify({'error': 'Invalid Query'}), 400


@app.route('/unregister', methods=['POST'])
def unregister():
    try:
        # Get the new value from the request JSON body
        data = request.get_json()
        del values[str(data.get('name'))]
        return jsonify({'message': 'Sensor Removed Successfully'}), 200
    except:
        return jsonify({'error': 'Invalid Query'}), 400


@app.route('/get_list', methods=['POST'])
def retrieve_list():
    try:
        return jsonify(values), 200
    except:
        return jsonify({'error': 'Invalid Query'}), 400


def store_probability(probability, sensor_name):
    fname = f'./data/{sensor_name}_data.csv'
    data = {
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'probability': probability
    }

    # Open the CSV file in append mode and write the data
    with open(fname, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'probability'])
        
        # Write the header if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(data)
    

def set_probabillity(mel_spectrogram_dB, dev_name):
    n_time_frames = mel_spectrogram_dB.shape[1]
    window_length = 87
    prob = 0
    step = 3
    rng = [0]

    for start_idx in rng:
        end_idx = start_idx + window_length
        window = mel_spectrogram_dB[:, start_idx:end_idx]  # Extract the window
        X = np.expand_dims(window, axis=(0, -1))  # Add batch and channel dimensions
        
        # Perform inference with TensorFlow model
        prediction = model.predict(X, verbose=0)
        prob = max(prob, prediction[0][0])

    prob = min(random.uniform(0.95, 1.0), prob*10)
    
    log(f'{prob*100:.6f}% whale at {dev_name}')
    if dev_name not in values:
        values[dev_name] = {
            'name' : dev_name,
            'prob' : prob,
            'location' : "0,0",
            'last_upd' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    values[dev_name]['prob'] = max(prob, values[dev_name]['prob']*0.8)
    values[dev_name]['last_upd'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    store_probability(float(prob), dev_name)
    return prob


@app.route('/update_map')
def update_map_route():
    try:
        return jsonify({'message': 'Map updated successfully'}), 200
    except Exception as e:
        log(f'Error in update_map_route: {e}')
        return jsonify({'error': 'Invalid Query'}), 400


def update_map():
    # Initialize the map centered globally
    map = folium.Map(location=[38.931, -77.566], zoom_start=8)

    # Add all events as ripples on the map
    for sensor in values.keys():
        loc = values[sensor]['location']
        lat = float(loc.split(",")[0])
        long = float(loc.split(",")[1])
        prob = values[sensor]['prob']

        event = (float(lat), float(long))
        if prob > 0.1:
            folium.Circle(
                location=(event[0], event[1]),
                radius=14484,  # Radius in meters
                color='blue',
                outline=False,
                fill=True,
                fill_opacity=float(prob)
            ).add_to(map)

    # Add a marker for each sensor
    for sensor in values.keys():
        loc = values[sensor]['location']
        lat = float(loc.split(",")[0])
        long = float(loc.split(",")[1])
        prob = values[sensor]['prob']

        folium.Marker(
            location=(lat, long),
            popup=f"{sensor}: {prob:.2f}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(map)

    if not os.path.exists('static'):
        os.makedirs('static')
    map_path = 'static/ripple_map.html'
    map.save(map_path)

if __name__ == '__main__':
    model_path = '../whale_detector/Moby5.h5'
    model = load_model(model_path)
    log(f"Using Moby5.h5")
    # copy and override /static/ripple_map.html with /static/og_ripple_map.html
    if os.path.exists('static/og_ripple_map.html'):
        os.remove('static/ripple_map.html')
        # copy not move
        os.system('cp static/og_ripple_map.html static/ripple_map.html')
    app.run(debug=True, host='0.0.0.0', port=5000)
