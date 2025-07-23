# prompt: use the test file in my drive for prediction of one row in my containerised flask app. set url of my cloud app to a var

import pandas as pd

# Send the prediction request to your Cloud Run service
import requests
import json
# Assuming the test file is named 'test.csv' and is in your Google Drive in the same directory as the training data.
# You'll need to adjust the path if it's elsewhere.
TEST_FILE_PATH = r"C:\Users\gohzh\OneDrive\Documents\School Stuff\04 AAP AI Applications Project\flask-prediction-service\test.csv"

# Assuming you want to predict for the first row of the test file.
try:
    test_df = pd.read_csv(TEST_FILE_PATH)
    # Select the first row and convert it to a dictionary for the API call
    # Drop 'id' if it exists in the test data and was dropped during training
    if 'id' in test_df.columns:
        test_df = test_df.drop(columns=['id'])
    first_row_dict = test_df.iloc[0].to_dict()

    # Your Cloud Run service URL
    CLOUD_APP_URL = "http://127.0.0.1:5000"  # Replace with your actual URL

    # Construct the full prediction endpoint URL for one row
    PREDICT_ONE_URL = f"{CLOUD_APP_URL}/predict-one"

    headers = {'Content-Type': 'application/json'}
    response = requests.post(PREDICT_ONE_URL, headers=headers, data=json.dumps(first_row_dict))

    if response.status_code == 200:
        prediction_result = response.json()
        print("Prediction for the first row:")
        print(prediction_result)
    else:
        print(f"Error predicting for the first row: {response.status_code} - {response.text}")

except FileNotFoundError:
    print(f"Error: Test file not found at {TEST_FILE_PATH}. Please check the path.")
except Exception as e:
    print(f"An error occurred during prediction request: {e}")
