from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

values = dict()

@app.route('/update', methods=['POST'])
def update_value():
    try:
        # Get the new value from the request JSON body
        data = request.get_json()
        new_value = float(data.get('prob'))
        dev_name = str(data.get('name'))
        values[dev_name]['prob'] = new_value
        values[dev_name]['last_upd'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({'message': 'Value updated successfully'}), 200
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
        throw
        return jsonify({'error': 'Invalid Query'}), 400


@app.route('/get_list', methods=['POST'])
def retrieve_list():
    try:
        return jsonify(values), 200
    except:
        throw
        return jsonify({'error': 'Invalid Query'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)