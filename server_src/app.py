import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
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
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import math
from models import db, User, Region, Sensor, Subscription, Notification

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mobyglobal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

values = dict()
spectrograms = dict()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")

@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/map')
def map_view():
    return render_template('map.html', active_page='map')

@app.route('/data')
def data_explorer():
    return render_template('data.html', active_page='data')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/try-it')
def try_it():
    return render_template('try_it.html', active_page='try-it')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validate form data
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return render_template('register.html', active_page='register')

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html', active_page='register')

        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html', active_page='register')

        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return render_template('register.html', active_page='register')

        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', active_page='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html', active_page='login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    # Get user's subscriptions
    subscriptions = Subscription.query.filter_by(user_id=current_user.id).all()

    # Get user's notifications
    notifications = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.created_at.desc()).limit(10).all()

    # Count unread notifications
    unread_count = Notification.query.filter_by(user_id=current_user.id, read=False).count()

    # Get available regions for subscription
    regions = Region.query.all()

    # Get available sensors for subscription
    sensors = Sensor.query.all()

    return render_template('profile.html',
                          active_page='profile',
                          subscriptions=subscriptions,
                          notifications=notifications,
                          unread_count=unread_count,
                          regions=regions,
                          sensors=sensors)

@app.route('/subscriptions')
@login_required
def subscriptions():
    # Get user's subscriptions
    user_subscriptions = Subscription.query.filter_by(user_id=current_user.id).all()

    # Get available regions for subscription
    regions = Region.query.all()

    # Get available sensors for subscription
    sensors = Sensor.query.all()

    return render_template('subscriptions.html',
                          active_page='subscriptions',
                          subscriptions=user_subscriptions,
                          regions=regions,
                          sensors=sensors)

@app.route('/subscribe', methods=['POST'])
@login_required
def subscribe():
    subscription_type = request.form.get('type')
    item_id = request.form.get('id')

    if not subscription_type or not item_id:
        flash('Invalid subscription request', 'danger')
        return redirect(url_for('subscriptions'))

    # Check if already subscribed
    if subscription_type == 'region':
        existing = Subscription.query.filter_by(user_id=current_user.id, region_id=item_id).first()
        if not existing:
            subscription = Subscription(user_id=current_user.id, region_id=item_id)
            db.session.add(subscription)
            region = Region.query.get(item_id)
            flash(f'Subscribed to {region.name} region', 'success')
    elif subscription_type == 'sensor':
        existing = Subscription.query.filter_by(user_id=current_user.id, sensor_id=item_id).first()
        if not existing:
            subscription = Subscription(user_id=current_user.id, sensor_id=item_id)
            db.session.add(subscription)
            sensor = Sensor.query.get(item_id)
            flash(f'Subscribed to {sensor.name} sensor', 'success')

    db.session.commit()
    return redirect(url_for('subscriptions'))

@app.route('/unsubscribe', methods=['POST'])
@login_required
def unsubscribe():
    subscription_id = request.form.get('subscription_id')

    if not subscription_id:
        flash('Invalid unsubscribe request', 'danger')
        return redirect(url_for('subscriptions'))

    subscription = Subscription.query.get(subscription_id)

    if subscription and subscription.user_id == current_user.id:
        db.session.delete(subscription)
        db.session.commit()
        flash('Unsubscribed successfully', 'success')

    return redirect(url_for('subscriptions'))

@app.route('/notifications')
@login_required
def notifications():
    page = request.args.get('page', 1, type=int)
    per_page = 20

    # Get user's notifications with pagination
    pagination = Notification.query.filter_by(user_id=current_user.id).order_by(
        Notification.created_at.desc()
    ).paginate(page=page, per_page=per_page)

    # Mark all as read
    unread = Notification.query.filter_by(user_id=current_user.id, read=False).all()
    for notification in unread:
        notification.read = True

    db.session.commit()

    return render_template('notifications.html',
                          active_page='notifications',
                          notifications=pagination.items,
                          pagination=pagination)

@app.route('/mark_notification_read/<int:notification_id>')
@login_required
def mark_notification_read(notification_id):
    notification = Notification.query.get(notification_id)

    if notification and notification.user_id == current_user.id:
        notification.read = True
        db.session.commit()

    return redirect(url_for('notifications'))

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Get the audio data from the request
        data = request.get_json()

        if 'audio_data' not in data:
            return jsonify({'error': 'No audio data provided'}), 400

        # Convert the audio data to numpy array
        audio_data = np.array(data.get('audio_data'), dtype=np.float32)

        # Log the audio data shape for debugging
        log(f"Received audio data with shape: {audio_data.shape}")

        # Generate mel spectrogram
        mel_spectrogram = get_melspectrogram(audio_data)

        # Log the spectrogram shape
        log(f"Generated spectrogram with shape: {mel_spectrogram.shape}")

        # Check if the spectrogram is large enough
        if mel_spectrogram.shape[1] < 87:
            log(f"Spectrogram too small: {mel_spectrogram.shape}")
            return jsonify({'error': 'Audio clip too short'}), 400

        # Process with sliding window approach
        n_time_frames = mel_spectrogram.shape[1]
        window_length = 87
        step = 10
        prob = 0

        # Calculate range for sliding window
        rng = range(0, n_time_frames - window_length + 1, step)

        # If range is empty (shouldn't happen given the check above), use at least the first window
        if len(rng) == 0:
            rng = [0]

        log(f"Processing {len(rng)} windows with step size {step}")

        # Process each window and keep the maximum probability
        for start_idx in rng:
            end_idx = start_idx + window_length
            window = mel_spectrogram[:, start_idx:end_idx]  # Extract the window
            X = np.expand_dims(window, axis=(0, -1))  # Add batch and channel dimensions

            # Perform inference with TensorFlow model
            prediction = model.predict(X, verbose=0)
            current_prob = float(prediction[0][0])
            prob = max(prob, current_prob)

            log(f"Window {start_idx}-{end_idx}: probability {current_prob*100:.2f}%")

        # Apply the same scaling as in set_probabillity function
        prob = min(random.uniform(0.95, 1.0), prob*10)

        log(f"Processed audio with probability: {prob*100:.2f}%")

        return jsonify({
            'probability': prob,
            'message': f'Whale detection probability: {prob*100:.2f}%',
            'processing_info': {
                'windows_processed': len(rng),
                'step_size': step,
                'window_length': window_length
            }
        }), 200

    except Exception as e:
        log(f'Error in process_audio: {e}')
        return jsonify({'error': str(e)}), 500


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

        if dev_name == "None":
            return jsonify({'error': 'Invalid Name'}), 400

        if dev_loc == "None":
            return jsonify({'error': 'Invalid Location'}), 400

        # Store in the values dictionary for backward compatibility
        values[dev_name] = {
            'name': dev_name,
            'prob': 0.0,
            'location': dev_loc,
            'last_upd': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        log(f'Register Data: {data}')

        # Check if sensor already exists in the database
        sensor = Sensor.query.filter_by(name=dev_name).first()

        if not sensor:
            # Parse location string to get latitude and longitude
            try:
                lat, lng = map(float, dev_loc.split(','))

                # Find the region this sensor belongs to
                region = None
                for r in Region.query.all():
                    # Calculate distance between sensor and region center
                    distance = calculate_distance(lat, lng, r.center_lat, r.center_lng)
                    if distance <= r.radius:
                        region = r
                        break

                # Create new sensor in database
                sensor = Sensor(
                    name=dev_name,
                    location=dev_loc,
                    region_id=region.id if region else None,
                    last_updated=datetime.now()
                )
                db.session.add(sensor)
                db.session.commit()

                # If sensor is in a region, notify subscribers
                if region:
                    notify_region_subscribers(region.id, f"New sensor '{dev_name}' added to {region.name}")
            except Exception as e:
                log(f"Error processing sensor location: {e}")

        return jsonify({'message': 'Sensor Registered Successfully'}), 200
    except Exception as e:
        log(f"Error in register_sensor: {e}")
        return jsonify({'error': 'Invalid Query'}), 400

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers using the Haversine formula"""
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r

def notify_region_subscribers(region_id, message):
    """Send notifications to all users subscribed to a region"""
    # Find all users subscribed to this region
    subscriptions = Subscription.query.filter_by(region_id=region_id).all()

    region = Region.query.get(region_id)
    if not region:
        return

    for subscription in subscriptions:
        # Create notification for each subscribed user
        notification = Notification(
            user_id=subscription.user_id,
            title=f"Update from {region.name}",
            message=message,
            region_id=region_id,
            created_at=datetime.now()
        )
        db.session.add(notification)

    db.session.commit()


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
    fname = f'./static/adddata/{sensor_name}_data.csv'
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

@app.route('/preview_csv/<filename>')
def preview_csv(filename):
    try:
        # Ensure the filename only contains safe characters
        if not all(c.isalnum() or c in '-_.' for c in filename):
            return "Invalid filename", 400

        # Determine the file path based on the filename
        if os.path.exists(f'./static/data/{filename}'):
            filepath = f'./static/data/{filename}'
        elif os.path.exists(f'./static/adddata/{filename}'):
            filepath = f'./static/adddata/{filename}'
        else:
            return "File not found", 404

        # Read the CSV file
        data = []
        headers = []
        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)  # Get headers

            # Limit to 1000 rows for preview
            row_count = 0
            for row in csv_reader:
                data.append(row)
                row_count += 1
                if row_count >= 1000:  # Limit to prevent large files from causing issues
                    break

        # Render the CSV data as HTML
        return render_template('csv_preview.html', filename=filename, headers=headers, data=data)

    except Exception as e:
        log(f'Error in preview_csv: {e}')
        return f"Error: {str(e)}", 500

@app.route('/graph_csv/<filename>')
def graph_csv(filename):
    """Render a template with a graph of the CSV data"""
    try:
        # Ensure the filename only contains safe characters
        if not all(c.isalnum() or c in '-_.' for c in filename):
            return "Invalid filename", 400

        # Check if file exists
        if os.path.exists(f'./static/data/{filename}'):
            return render_template('csv_graph.html', filename=filename)
        elif os.path.exists(f'./static/adddata/{filename}'):
            return render_template('csv_graph.html', filename=filename)
        else:
            return "File not found", 404
    except Exception as e:
        log(f'Error in graph_csv: {e}')
        return f"Error: {str(e)}", 500

@app.route('/api/csv_data/<filename>')
def csv_data(filename):
    """API endpoint to get CSV data in JSON format for charting"""
    try:
        # Ensure the filename only contains safe characters
        if not all(c.isalnum() or c in '-_.' for c in filename):
            return jsonify({"error": "Invalid filename"}), 400

        # Determine the file path based on the filename
        if os.path.exists(f'./static/data/{filename}'):
            filepath = f'./static/data/{filename}'
        elif os.path.exists(f'./static/adddata/{filename}'):
            filepath = f'./static/adddata/{filename}'
        else:
            return jsonify({"error": "File not found"}), 404

        # Read the CSV file
        timestamps = []
        values = []
        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header

            # Limit to 10000 data points for performance
            for i, row in enumerate(csv_reader):
                if i >= 10000:
                    break

                if len(row) >= 2:  # Ensure row has enough columns
                    timestamps.append(row[0])
                    values.append(float(row[1]))

        return jsonify({
            "timestamps": timestamps,
            "values": values
        })

    except Exception as e:
        log(f'Error in csv_data: {e}')
        return jsonify({"error": str(e)}), 500

def set_probabillity(mel_spectrogram_dB, dev_name):
    window_length = 87
    prob = 0
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

    # Get previous probability if it exists
    prev_prob = 0
    if dev_name in values:
        prev_prob = values[dev_name]['prob']

    # Create sensor entry if it doesn't exist
    if dev_name not in values:
        values[dev_name] = {
            'name': dev_name,
            'prob': prob,
            'location': "0,0",
            'last_upd': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # Update probability and timestamp
    values[dev_name]['prob'] = max(prob, values[dev_name]['prob']*0.8)
    values[dev_name]['last_upd'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Store probability in CSV
    store_probability(float(prob), dev_name)

    # Check if this is a significant whale detection (probability > 0.5)
    # and if it's a new detection or significantly higher than before
    if prob > 0.5 and (prev_prob < 0.3 or prob > prev_prob * 1.5):
        # Find the sensor in the database
        sensor = Sensor.query.filter_by(name=dev_name).first()

        if sensor:
            # Notify users subscribed to this specific sensor
            notify_sensor_subscribers(sensor.id, f"High whale detection probability ({prob*100:.1f}%) at {dev_name}")

            # If sensor is in a region, notify region subscribers too
            if sensor.region_id:
                region = Region.query.get(sensor.region_id)
                if region:
                    notify_region_subscribers(
                        region.id,
                        f"High whale detection probability ({prob*100:.1f}%) at {dev_name} in {region.name}"
                    )

    return prob

def notify_sensor_subscribers(sensor_id, message):
    """Send notifications to all users subscribed to a specific sensor"""
    # Find all users subscribed to this sensor
    subscriptions = Subscription.query.filter_by(sensor_id=sensor_id).all()

    sensor = Sensor.query.get(sensor_id)
    if not sensor:
        return

    for subscription in subscriptions:
        # Create notification for each subscribed user
        notification = Notification(
            user_id=subscription.user_id,
            title=f"Alert from {sensor.name}",
            message=message,
            sensor_id=sensor_id,
            created_at=datetime.now()
        )
        db.session.add(notification)

    db.session.commit()


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

def init_db():
    """Initialize the database with default regions if they don't exist"""
    with app.app_context():
        # Create all tables
        db.create_all()

        # Check if we need to add default regions
        if Region.query.count() == 0:
            # Add some default regions
            default_regions = [
                {
                    'name': 'North Atlantic',
                    'description': 'North Atlantic Ocean region',
                    'center_lat': 45.0,
                    'center_lng': -40.0,
                    'radius': 2000.0
                },
                {
                    'name': 'North Pacific',
                    'description': 'North Pacific Ocean region',
                    'center_lat': 40.0,
                    'center_lng': -150.0,
                    'radius': 2000.0
                },
                {
                    'name': 'South Pacific',
                    'description': 'South Pacific Ocean region',
                    'center_lat': -20.0,
                    'center_lng': -120.0,
                    'radius': 2000.0
                },
                {
                    'name': 'Indian Ocean',
                    'description': 'Indian Ocean region',
                    'center_lat': -10.0,
                    'center_lng': 80.0,
                    'radius': 2000.0
                },
                {
                    'name': 'Arctic',
                    'description': 'Arctic Ocean region',
                    'center_lat': 80.0,
                    'center_lng': 0.0,
                    'radius': 1500.0
                },
                {
                    'name': 'Southern Ocean',
                    'description': 'Southern Ocean around Antarctica',
                    'center_lat': -70.0,
                    'center_lng': 0.0,
                    'radius': 2000.0
                }
            ]

            for region_data in default_regions:
                region = Region(**region_data)
                db.session.add(region)

            db.session.commit()
            log("Added default regions to database")

        # Import existing sensors into the database
        if Sensor.query.count() == 0 and values:
            for sensor_name, sensor_data in values.items():
                # Check if sensor already exists
                if not Sensor.query.filter_by(name=sensor_name).first():
                    try:
                        location = sensor_data.get('location', '0,0')

                        # Try to find the region this sensor belongs to
                        region_id = None
                        try:
                            lat, lng = map(float, location.split(','))

                            for region in Region.query.all():
                                distance = calculate_distance(lat, lng, region.center_lat, region.center_lng)
                                if distance <= region.radius:
                                    region_id = region.id
                                    break
                        except:
                            pass

                        # Create the sensor
                        sensor = Sensor(
                            name=sensor_name,
                            location=location,
                            region_id=region_id,
                            last_updated=datetime.now()
                        )
                        db.session.add(sensor)
                    except Exception as e:
                        log(f"Error importing sensor {sensor_name}: {e}")

            db.session.commit()
            log("Imported existing sensors to database")

if __name__ == '__main__':
    model_path = '../whale_detector/Moby5.h5'
    model = load_model(model_path)
    log(f"Using Moby5.h5")

    # Initialize the database
    init_db()

    # Copy and override /static/ripple_map.html with /static/og_ripple_map.html
    if os.path.exists('static/og_ripple_map.html'):
        os.remove('static/ripple_map.html')
        # copy not move
        os.system('cp static/og_ripple_map.html static/ripple_map.html')

    app.run(debug=True, host='0.0.0.0', port=5002)
