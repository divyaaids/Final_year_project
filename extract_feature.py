import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

try:
    df = pd.read_csv('dataset/preprocessed_data.csv')
    logging.info("Preprocessed data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading preprocessed_data.csv: {e}")
    exit()

# Calculate additional features
df['PPG_to_HR'] = df['PPG_Signal'] / (df['Heart_Rate'] + 1e-5)
df['PPG_HR_Product'] = df['PPG_Signal'] * df['Heart_Rate']

# Save features
df.to_csv('dataset/features_data.csv', index=False)
logging.info("Feature extraction complete. Data saved to 'features_data.csv'.")
