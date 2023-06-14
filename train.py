"""
This script is used to train a RandomForestRegressor model for forex data prediction.
It iterates over all .csv files in the 'data' directory, each file represents a specific 
currency pair. For each currency pair, the script performs the following operations:

1. Read and Load the data
2. Convert timestamp to datetime format
3. Sort data by timestamp
4. Extract time-based features (hour, day_of_week, month)
5. Prepare the data for training (handle missing values, define features and targets)
6. Split the data into a training and testing set
7. Scale the features for better model performance
8. Train the RandomForestRegressor model
9. Make predictions on the testing set and calculate the Root Mean Squared Error (RMSE)
10. Save both the trained model and the scaler for future use

In order to use this script, make sure that:
- You have the necessary dependencies installed (python3, pandas, sklearn, joblib, os, logging).
- Your data files are in the correct format and placed in the 'data' directory.
- The 'trained_data' directory exists for model saving.

To run the script, simply execute: python train.py

Note: This script assumes that the target variable is the 'close' price. If your target 
variable is different, please adjust the target variable in the code accordingly.
"""
import os, joblib, logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger('Training')

data_dir = 'data'
data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

for data_file in data_files:

    currency_pair = os.path.splitext(data_file)[0]
    filepath = os.path.join(data_dir, data_file)

    logger.info(f"[{currency_pair}] Training")

    logger.info(f"[{currency_pair}] Reading dataset...")
    columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    data = pd.read_csv(filepath, header=None, names=columns)
    logger.info(f"[{currency_pair}] Dataset loaded.")

    logger.info(f"[{currency_pair}] Converting timestamp to datetime...")
    data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%Y.%m.%d %H:%M')
    logger.info(f"[{currency_pair}] Conversion completed.")

    logger.info(f"[{currency_pair}] Sorting data by timestamp...")
    data = data.sort_values(by='timestamp')
    logger.info(f"[{currency_pair}] Sorting completed.")

    logger.info(f"[{currency_pair}] Extracting time-based features...")
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['month'] = data['timestamp'].dt.month
    logger.info(f"[{currency_pair}] Time-based feature extraction completed.")

    logger.info(f"[{currency_pair}] Training model...")

    logger.info(f"[{currency_pair}] Dropping the last rows with NaNs...")
    data_clean = data.dropna()  # Drop the rows with NaNs in the target column
    data_clean = data_clean[:-1]
    logger.info(f"[{currency_pair}] Dropping complete.")

    logger.info(f"[{currency_pair}] Defining features and target...")
    features = data_clean.drop(columns=['timestamp', 'date', 'time'])
    logger.info(f"[{currency_pair}] Definition of features and target completed.")

    target = features['close']

    # Instead of a random train-test split, for time series data we should use a continuous split
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

    logger.info(f"[{currency_pair}] Scaling features...")
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)
    logger.info(f"[{currency_pair}] Scaling of features completed.")

    logger.info(f"[{currency_pair}] Training the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(features_train_scaled, target_train)
    logger.info(f"[{currency_pair}] Model training completed.")

    logger.info(f"[{currency_pair}] Making predictions on the test set...")
    predictions = model.predict(features_test_scaled)
    logger.info(f"[{currency_pair}] Prediction completed.")

    logger.info(f"[{currency_pair}] Calculating RMSE...")
    rmse = mean_squared_error(target_test, predictions, squared=False)
    logger.info(f'[{currency_pair}] Test RMSE: {rmse}')

    logger.info(f"[{currency_pair}] Saving the model and scaler...")
    joblib.dump(model, f'trained_data/model/{currency_pair}.joblib')  # Save the model
    joblib.dump(scaler, f'trained_data/scalar/{currency_pair}.joblib')  # Save the scaler
    logger.info(f"[{currency_pair}] Model and scaler saved successfully.")

    logger.info(f"[{currency_pair}] Training Complete")

logger.info('Training complete')