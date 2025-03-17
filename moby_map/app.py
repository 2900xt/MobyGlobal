from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
import os

app = Flask(__name__)
events = []

@app.route('/')
def index():
    return render_template('map.html')

@app.route('/add_event', methods=['POST'])
def add_event():
    data = request.get_json()
    lat = data['lat']
    long = data['long']
    events.append((lat, long))
    return jsonify({"status": "success"})

@app.route('/update_map')
def update_map():
    # Initialize the map centered globally
    map = folium.Map(location=[0, 0], zoom_start=2)

    # Add all events as ripples on the map
    for event in events:
        folium.Circle(
            location=(event[0], event[1]),
            radius=50000,  # Radius in meters
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(map)

    if not os.path.exists('static'):
        os.makedirs('static')
    map_path = 'static/ripple_map.html'
    map.save(map_path)
    return jsonify({"status": "map updated"})

@app.route('/static/ripple_map.html')
def ripple_map():
    return send_from_directory('static', 'ripple_map.html')

if __name__ == "__main__":
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, host='127.0.0.0', port=5001)