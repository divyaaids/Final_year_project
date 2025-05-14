from flask import Flask, render_template, jsonify
import serial
import serial.tools.list_ports
import threading
import time
import pickle
import re
import pandas as pd
import logging
import configparser
import os

app = Flask(__name__)

DATA_FILE = 'sensor_data.ini'
CONFIG_SECTION = 'SENSORS'

# Lock for safe concurrent access
data_lock = threading.Lock()

model = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------
# Load / Save INI Helpers
# -----------------------

def load_sensor_data():
    config = configparser.ConfigParser()
    if os.path.exists(DATA_FILE):
        config.read(DATA_FILE)
        if CONFIG_SECTION in config:
            return {
                'ppg_signal': config[CONFIG_SECTION].get('ppg_signal', 'N/A'),
                'heart_rate': config[CONFIG_SECTION].get('heart_rate', 'N/A'),
                'glucose_prediction': config[CONFIG_SECTION].get('glucose_prediction', 'N/A'),
                'suggestion': config[CONFIG_SECTION].get('suggestion', 'Waiting for data...')
            }
    return {
        'ppg_signal': 'N/A',
        'heart_rate': 'N/A',
        'glucose_prediction': 'N/A',
        'suggestion': 'Waiting for data...'
    }

def save_sensor_data(data):
    config = configparser.ConfigParser()
    config[CONFIG_SECTION] = {
        'ppg_signal': str(data['ppg_signal']),
        'heart_rate': str(data['heart_rate']),
        'glucose_prediction': str(data['glucose_prediction']),
        'suggestion': data['suggestion']
    }
    with open(DATA_FILE, 'w') as configfile:
        config.write(configfile)

# -----------------------
# Serial + Model Functions
# -----------------------

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description:
            return port.device
    return "COM3"

def load_model():
    global model
    try:
        with open('model/glucose_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

def serial_read_thread():
    port = find_arduino_port()
    if port == "COM3":
        logging.warning("No Arduino detected automatically. Defaulting to COM3.")

    try:
        ser = serial.Serial(port, 9600, timeout=2)
        logging.info(f"Connected to Arduino on {port}")
        time.sleep(2)
    except Exception as e:
        logging.error(f"Serial error: {e}")
        return

    pattern = re.compile(r'\*(\d+)\*(\d+)#')

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                match = pattern.match(line)
                if match:
                    ppg = int(match.group(1))
                    hr = int(match.group(2))

                    features = pd.DataFrame([[ppg, hr, ppg / (hr + 1e-5), ppg * hr]],
                                            columns=['PPG_Signal', 'Heart_Rate', 'PPG_to_HR', 'PPG_HR_Product'])

                    prediction = model.predict(features)[0]

                    if prediction < 80:
                        sug = "Glucose low. Consume carbohydrates."
                    elif prediction > 140:
                        sug = "Glucose high. Seek medical advice."
                    else:
                        sug = "Glucose normal."

                    with data_lock:
                        sensor_data = {
                            'ppg_signal': ppg,
                            'heart_rate': hr,
                            'glucose_prediction': round(prediction, 2),
                            'suggestion': sug
                        }
                        save_sensor_data(sensor_data)

                    logging.info(f"Data saved: PPG={ppg}, HR={hr}, Prediction={prediction:.2f}")
        except Exception as e:
            logging.error(f"Error reading serial data: {e}")
        time.sleep(0.1)

# -----------------------
# Flask Routes
# -----------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    with data_lock:
        sensor_data = load_sensor_data()
        return jsonify({
            'PPG_Signal': str(sensor_data['ppg_signal']),
            'Heart_Rate': str(sensor_data['heart_rate']),
            'Glucose_Prediction': str(sensor_data['glucose_prediction']),
            'Suggestion': sensor_data['suggestion']
        })

# -----------------------
# Main Entry
# -----------------------

if __name__ == '__main__':
    load_model()
    thread = threading.Thread(target=serial_read_thread, daemon=True)
    thread.start()
    app.run(debug=True, host='0.0.0.0')
