import serial
import serial.tools.list_ports
import csv
import time
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description:
            return port.device
    return "COM3"

arduino_port = find_arduino_port()
if arduino_port:
    with open("PORT", "w") as file:
        file.write(arduino_port)
    logging.info(f"Arduino found on {arduino_port} and saved to PORT.")
else:
    logging.error("No Arduino detected.")
    exit()

try:
    ser = serial.Serial(arduino_port, 9600, timeout=2)
    logging.info(f"Connected to Arduino on {arduino_port}.")
    time.sleep(2)  # Wait for Arduino to reset
except Exception as e:
    logging.error(f"Serial Error: {e}")
    exit()

CSV_FILE = "dataset/data.csv"
pattern = re.compile(r'\*(\d+)\*(\d+)#')

def append_data():
    try:
        # Optionally send a command to Arduino (if needed)
        ser.write(b"send\n")
        time.sleep(1)
        data = ser.readline().decode("utf-8").strip()
        match = pattern.match(data)
        if match:
            ppg = int(match.group(1))
            hr = int(match.group(2))
            glucose_input = input("Enter measured Glucose Level: ").strip()
            try:
                glucose = float(glucose_input)
                with open(CSV_FILE, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ppg, hr, glucose])
                logging.info(f"Data appended: PPG={ppg}, HR={hr}, Glucose={glucose}")
            except ValueError:
                logging.error("Invalid glucose value entered.")
        else:
            logging.error("Incorrect data format received from Arduino.")
    except Exception as e:
        logging.error(f"Error during data append: {e}")

print("Press 'a' to append data or 'q' to quit.")
while True:
    user_input = input("Command (a/q): ").strip().lower()
    if user_input == "q":
        logging.info("Exiting data append script.")
        break
    elif user_input == "a":
        append_data()
    else:
        logging.warning("Invalid input! Please enter 'a' or 'q'.")

ser.close()
