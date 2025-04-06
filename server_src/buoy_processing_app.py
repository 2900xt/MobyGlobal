from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
import os
import numpy as np
import json
import csv
from datetime import datetime
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import requests

app = Flask(__name__)
values = dict()

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")

@app.route('/')
def index():
    return render_template('map.html')


@app.route('/update', methods=['POST'])
def update_data():
    try:
        # Get the new value from the request JSON body
        data = request.get_json()
        if 'probabillity' not in data:
            return jsonify({'error': 'No value provided'}), 400
        
        if 'name' not in data:
            return jsonify({'error': 'No name provided'}), 400

        prob = np.array(data.get('probabillity'))
        dev_name = str(data.get('name'))
        prob = set_probabillity(dev_name, prob)

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
        return jsonify({'message': 'Sensor Registered Successfully'}), 200
    except Exception:
        throw
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
    

def set_probabillity(dev_name, prob):
    # n_time_frames = mel_spectrogram_dB.shape[1]
    # window_length = 87
    # prob = 0
    # step = 3
    # rng = range(0, n_time_frames - window_length + 1, step)

    # for start_idx in rng:
    #     end_idx = start_idx + window_length
    #     window = mel_spectrogram_dB[:, start_idx:end_idx]  # Extract the window
    #     X = np.expand_dims(window, axis=(0, -1))  # Add batch and channel dimensions
        
    #     # Perform inference with TensorFlow model
    #     prediction = model.predict(X, verbose=0)
    #     prob = max(prob, prediction[0][0])

    # prob = min(random.uniform(0.95, 1.0), prob*10)
    
    log(f'{prob*100:.6f}% whale at {dev_name}')
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
        print(f"Adding {sensor} to map at {lat}, {long} with probability {prob}")

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
        print(f"Adding {sensor} to map at {lat}, {long} with probability {prob}")

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
    model_path = '/home/taha/rsef2025/whale_detector/Moby5.h5'
    model = load_model(model_path)
    log(f"Using Moby5.h5")
    # copy and override /static/ripple_map.html with /static/og_ripple_map.html
    if os.path.exists('static/og_ripple_map.html'):
        os.remove('static/ripple_map.html')
        # copy not move
        os.system('cp static/og_ripple_map.html static/ripple_map.html')
    app.run(debug=True, host='0.0.0.0', port=5000)