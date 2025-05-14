import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pickle
import logging
import time
import os

# Configure logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths for dataset and model storage
data_path = 'dataset/features_data.csv'
model_dir = 'model'
model_path = os.path.join(model_dir, 'glucose_model.pkl')

# Create model directory if it does not exist
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
    logging.info(f"Created model directory at: {model_dir}")

# Load dataset with error handling
try:
    df = pd.read_csv(data_path)
    logging.info(f"Features data loaded successfully from {data_path}. Data shape: {df.shape}")
except Exception as e:
    logging.error(f"Error loading {data_path}: {e}")
    exit()

logging.info(f"Dataset columns: {df.columns.tolist()}")
logging.info(f"First 5 rows:\n{df.head()}")

# Define features and target variable
feature_columns = ['PPG_Signal', 'Heart_Rate', 'PPG_to_HR', 'PPG_HR_Product']
target_column = 'Glucose_level'
X = df[feature_columns]
y = df[target_column]

logging.info("Features and target variable set.")
logging.info(f"Features used: {feature_columns}")
logging.info(f"Target: {target_column}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Data split into training and testing sets.")
logging.info(f"Training set size: {X_train.shape[0]} samples")
logging.info(f"Testing set size: {X_test.shape[0]} samples")

# Use a small random subset (e.g., 10%) of training data for hyperparameter tuning
sample_fraction = 0.1
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train, test_size=(1 - sample_fraction), random_state=42
)
logging.info(f"Using a random subset of {X_train_sample.shape[0]} samples for hyperparameter tuning.")

# Define hyperparameter grid
param_dist = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20]
}

# Initialize RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Set up RandomizedSearchCV for faster hyperparameter tuning
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=6, cv=3, scoring='neg_mean_squared_error',
                                   verbose=1, n_jobs=-1, random_state=42)

# Time the hyperparameter search process on the sample data
logging.info("Starting hyperparameter tuning with RandomizedSearchCV...")
start_time = time.time()
random_search.fit(X_train_sample, y_train_sample)
elapsed_time = time.time() - start_time
logging.info(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds.")

# Retrieve the best parameters found
best_params = random_search.best_params_
logging.info(f"Best hyperparameters found on sample: {best_params}")

# Retrain the model on the full training data using the best parameters
logging.info("Retraining model on full training data with the best hyperparameters...")
best_model = RandomForestRegressor(**best_params, random_state=42)
start_time = time.time()
best_model.fit(X_train, y_train)
elapsed_time = time.time() - start_time
logging.info(f"Final model training completed in {elapsed_time:.2f} seconds on full training data.")

# Evaluate the final model on the testing set
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
logging.info(f"Final model performance on test set - Mean Squared Error (MSE): {mse:.2f}")

# Optionally display a sample of true vs. predicted values
sample_results = pd.DataFrame({
    'True Glucose': y_test.values,
    'Predicted Glucose': predictions
})
logging.info(f"Sample predictions:\n{sample_results.head()}")

# Save the final trained model to disk
try:
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logging.info(f"Trained model saved successfully at: {model_path}")
except Exception as e:
    logging.error(f"Error saving model at {model_path}: {e}")
