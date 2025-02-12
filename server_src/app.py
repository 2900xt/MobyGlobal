from flask import Flask, request, jsonify
import numpy as np
import json
import csv
from datetime import datetime
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
values = dict()

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")


@app.route('/update', methods=['POST'])
def update_data():
    try:
        # Get the new value from the request JSON body
        data = request.get_json()
        if 'data' not in data:
            return jsonify({'error': 'No array provided'}), 400
        
        if 'name' not in data:
            return jsonify({'error': 'No name provided'}), 400

        new_spectogram = np.array(data.get('data'))
        print(len(new_spectogram))
        dev_name = str(data.get('name'))
        prob = set_probabillity(new_spectogram, dev_name)

        return jsonify({'message': f'Data updated successfully. P(whale) = {prob}'}), 200
    except:
        throw
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
    

def set_probabillity(mel_spectrogram_dB, dev_name):
    n_time_frames = mel_spectrogram_dB.shape[1]
    window_length = 87
    prob = 0
    step = 3
    rng = range(0, n_time_frames - window_length + 1, step)

    for start_idx in rng:
        end_idx = start_idx + window_length
        window = mel_spectrogram_dB[:, start_idx:end_idx]  # Extract the window
        X = np.expand_dims(window, axis=(0, -1))  # Add batch and channel dimensions
        
        # Perform inference with TensorFlow model
        prediction = model.predict(X, verbose=0)
        prob = max(prob, prediction[0][0])
    
    log(f'{prob*100:.6f}% whale at {dev_name}')
    values[dev_name]['prob'] = prob
    values[dev_name]['last_upd'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    store_probability(float(prob), dev_name)
    return prob

if __name__ == '__main__':
    model_path = '/home/taha/rsef2025/whale_detector/Moby5.h5'
    model = load_model(model_path)
    log(f"Using Moby5.h5")
    app.run(debug=True)