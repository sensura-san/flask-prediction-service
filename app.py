# prompt: convert above flask into containerisable single file

import numpy as np
import pandas as pd
# import math
# import matplotlib.pyplot as plt
# import seaborn as sns
import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import compute_class_weight
# from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, confusion_matrix
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
from flask import Flask, request, jsonify
# Assuming the dataset is in a file named 'train.csv' in the same directory
# If running in Colab, you might need to adjust the path or mount Drive/use GCS

# Define the feature engineering function
def feature_engineering(data):
    # Check if 'Length_of_Conveyer' and 'Steel_Plate_Thickness' exist before division
    if 'Length_of_Conveyer' in data.columns and 'Steel_Plate_Thickness' in data.columns and (data['Steel_Plate_Thickness'] != 0).all():
         data['Ratio_Length_Thickness'] = data['Length_of_Conveyer'] / data['Steel_Plate_Thickness']
    else:
        # Handle cases where columns are missing or Steel_Plate_Thickness is zero
        # You might want to fill with a default value or NaN
        data['Ratio_Length_Thickness'] = np.nan # or some other handling

    # Check if 'Steel_Plate_Thickness' exists before normalization
    if 'Steel_Plate_Thickness' in data.columns:
        min_thickness = data['Steel_Plate_Thickness'].min()
        max_thickness = data['Steel_Plate_Thickness'].max()
        if max_thickness != min_thickness:
             data['Normalized_Steel_Thickness'] = (data['Steel_Plate_Thickness'] - min_thickness) / (max_thickness - min_thickness)
        else:
             data['Normalized_Steel_Thickness'] = 0 # Or some other handling if all values are the same
    else:
        data['Normalized_Steel_Thickness'] = np.nan # Or some other handling


    # Check if 'X_Maximum', 'X_Minimum', and 'Pixels_Areas' exist
    if all(col in data.columns for col in ['X_Maximum', 'X_Minimum', 'Pixels_Areas']):
        data['X_Range*Pixels_Areas'] = (data['X_Maximum'] - data['X_Minimum']) * data['Pixels_Areas']
    else:
        data['X_Range*Pixels_Areas'] = np.nan # Or some other handling

    return data

# List of features to drop based on EDA/Feature Selection
FEATURES_TO_DROP = ['Y_Minimum', 'Steel_Plate_Thickness', 'Sum_of_Luminosity', 'Edges_X_Index', 'SigmoidOfAreas', 'Luminosity_Index', 'TypeOfSteel_A300']

# Define the mapping from predicted class index to label
TARGET_FEATURES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
label_mapping = {i + 1: TARGET_FEATURES[i] for i in range(len(TARGET_FEATURES))}
label_mapping[0] = 'No_Defect'
# Ensure the mapping is ordered by key
sorted_labels = sorted(label_mapping.keys())
display_labels = [label_mapping[key] for key in sorted_labels]

# Load the trained models and scaler
# Ensure the model file 'all_deployment_artifacts.joblib' is accessible.
# In a container, you'd ensure this file is copied into the image.
ARTIFACTS_FILE = 'all_deployment_artifacts.joblib'

try:
    artifacts = joblib.load(ARTIFACTS_FILE)
    # scaler = artifacts['scaler']
    # Load the models you want to make available for prediction
    # For the API, let's load the 'cat' model as indicated as the best
    imported_model = artifacts['cat'] # or choose another model
except FileNotFoundError:
    print(f"Error: Model file '{ARTIFACTS_FILE}' not found.")
    # In a real application, you would handle this error more robustly,
    # perhaps exiting the application or raising an exception.
    # scaler = None
    imported_model = None
except KeyError as e:
     print(f"Error: Missing expected key in artifact file: {e}")
    #  scaler = None
     imported_model = None


# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Steel Plate Defect Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    if imported_model is None:
        return jsonify({'error': 'Model or scaler not loaded. Check server logs.'}), 500

    try:
        # Get data from the request
        data = request.get_json(force=True)

        # Convert the input data to a pandas DataFrame
        # Handle cases where input is a single dictionary vs. a list of dictionaries
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid input data format. Expected dictionary or list of dictionaries.'}), 400

        # Apply the same feature engineering steps used during training
        processed_df = feature_engineering(input_df.copy()) # Use a copy to avoid modifying the original input

        # Drop the same features that were dropped during training
        # Use errors='ignore' in case the input data is already pre-processed
        processed_df = processed_df.drop(FEATURES_TO_DROP, axis=1, errors='ignore')

        # Ensure the order of columns matches the training data
        # This is crucial for consistent scaling and prediction
        # You would need the list of training columns (X.columns) from your training phase.
        # Assuming X.columns was available globally or saved in artifacts.
        # For this single file, let's assume X.columns was derived from train.csv
        # A more robust approach would save this list in the artifacts file.

        # --- BEGIN: Placeholder for getting training columns ---
        # In a real deployment, you would load the list of columns that were
        # used to train the model. For this example, let's load the training data
        # again just to get the column order after processing. This is not ideal
        # for production as it ties the API to the training data file.
        try:
            # Load the training data again to get the column order
            train_df_for_cols = pd.read_csv("train.csv") # Adjust path if necessary
            train_df_for_cols.drop(columns=['id'] + TARGET_FEATURES, inplace=True, errors='ignore')
            train_df_for_cols = feature_engineering(train_df_for_cols)
            train_cols_order = train_df_for_cols.drop(FEATURES_TO_DROP, axis=1, errors='ignore').columns.tolist()
        except FileNotFoundError:
            return jsonify({'error': "Could not load training data to determine column order."}), 500
        except Exception as e:
             return jsonify({'error': f"Error processing training data for column order: {e}"}), 500
        # --- END: Placeholder ---

        # Reindex the input data to match the training column order
        # Fill missing columns with NaN if they were not present in the input
        try:
             processed_df = processed_df.reindex(columns=train_cols_order, fill_value=np.nan)
        except Exception as e:
            return jsonify({'error': f"Error reindexing input columns: {e}"}), 400


        # Check for any remaining NaNs after reindexing that were not handled by feature engineering
        # Depending on your model, you might need to impute NaNs here.
        # For simplicity, let's assume the input data is clean or features engineering handled it.
        if processed_df.isnull().sum().sum() > 0:
            print("Warning: NaNs found in processed input data. Imputation might be needed.")
            # You might want to impute NaNs here, e.g., using `processed_df.fillna(0, inplace=True)`
            # or using an imputer saved during training.


        # Scale the input data using the fitted scaler
        # try:
        #     input_scaled = scaler.transform(processed_df)
        # except Exception as e:
        #      return jsonify({'error': f"Error scaling input data: {e}"}), 400


        # Make prediction
        predictions_proba = imported_model.predict_proba(processed_df)
        predictions_indices = imported_model.predict(processed_df)

        # Convert predicted indices to labels
        predictions_labels = [label_mapping.get(idx, 'Unknown') for idx in predictions_indices]

        # Format the output
        results = []
        for i in range(len(input_df)):
            sample_result = {
                'predicted_label': predictions_labels[i],
                'predicted_class_index': int(predictions_indices[i]), # Convert numpy int to Python int
                'probabilities': {display_labels[j]: float(predictions_proba[i][j]) for j in range(predictions_proba.shape[1])}
            }
            results.append(sample_result)

        return jsonify(results)

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # When running in a container, you'll typically expose a specific port.
    # The default Flask port is 5000.
    # In a Dockerfile, you would use EXPOSE 5000.
    # When running the container, you would map a host port to container port 5000.
    print("Starting Flask server...")
    # Use 0.0.0.0 to make the server accessible from outside the container/localhost
    # In a production environment, use a proper WSGI server (like Gunicorn or uWSGI)
    app.run(host='0.0.0.0', port=5000)
