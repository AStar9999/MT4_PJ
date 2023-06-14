# Forex Prediction Model Training - README.md

## Overview
This project involves training a RandomForestRegressor model for predicting forex prices. The training script (train.py) takes .csv data files for different currency pairs, performs various preprocessing steps, trains the model, makes predictions on a test set, and finally, saves the model and scaler for future use.

## Prerequisites
To run this project, you'll need the following installed:
- Python 3
- Pip (Python package manager)

We recommend using a virtual environment (venv) to install the necessary dependencies, to avoid conflicts with packages in your global Python environment.

## Installation
1. Clone this repository:
    ```
    git clone git@github.com:AStar9999/MT4_PJ.git
    ```

2. (Recommended) Create a virtual environment:
    ```
    python3 -m venv venv
    ```

3. Activate the virtual environment:

    - On macOS and Linux:
        ```
        source venv/bin/activate
        ```

    - On Windows:
        ```
        .\venv\Scripts\activate
        ```

4. Install the necessary dependencies:
    ```
    pip install -r requirements.txt
    ```

## Running the Training Script
Once you've installed the dependencies, you can run the training script with the following command:

```
python3 train.py
```

The script will read in .csv files from the 'data' directory, each representing a specific currency pair. It will perform preprocessing steps including time-based feature extraction and scaling, then train a RandomForestRegressor model on the data. The model and the scaler will be saved in the 'trained_data' directory for future use.

Note: The script assumes that the target variable is the 'close' price. If your target variable is different, adjust the target variable in the code accordingly. Also, remember that this script uses StandardScaler from sklearn for feature scaling. This affects how the data is processed and how the model is trained.

## Data
The .csv data files should be placed in the 'data' directory. The script expects the files to be in the following format:

```
date,time,open,high,low,close,volume
```

The timestamp is converted to datetime format and the data is sorted by timestamp during preprocessing.

## Model and Scaler
After training, the RandomForestRegressor model and the StandardScaler are saved in the 'trained_data/model' and 'trained_data/scaler' directories respectively. These can be loaded in future for making predictions on new data.

## Additional Information
For more details on the training process, refer to the top-level comments in the 'train.py' script.