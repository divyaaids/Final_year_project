import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

try:
    df = pd.read_csv('dataset/data.csv')
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading data.csv: {e}")
    exit()

# Separate features and target
features = df[['PPG_Signal', 'Heart_Rate']]
target = df['Glucose_level']

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
logging.info("Features normalized using MinMaxScaler.")

# Create a DataFrame with normalized features
df_scaled = pd.DataFrame(features_scaled, columns=['PPG_Signal', 'Heart_Rate'])
df_scaled['Glucose_level'] = target

# Save the preprocessed data
df_scaled.to_csv('dataset/preprocessed_data.csv', index=False)
logging.info("Preprocessed data saved to 'preprocessed_data.csv'.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

train_df = pd.DataFrame(X_train, columns=['PPG_Signal', 'Heart_Rate'])
train_df['Glucose_level'] = y_train.values
train_df.to_csv('dataset/train_data.csv', index=False)

test_df = pd.DataFrame(X_test, columns=['PPG_Signal', 'Heart_Rate'])
test_df['Glucose_level'] = y_test.values
test_df.to_csv('dataset/test_data.csv', index=False)

logging.info("Data split complete. 'train_data.csv' and 'test_data.csv' created.")
