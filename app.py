import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Define the feature engineering function
def feature_engineering(data):
    if 'Length_of_Conveyer' in data.columns and 'Steel_Plate_Thickness' in data.columns and (data['Steel_Plate_Thickness'] != 0).all():
        data['Ratio_Length_Thickness'] = data['Length_of_Conveyer'] / data['Steel_Plate_Thickness']
    else:
        data['Ratio_Length_Thickness'] = np.nan

    if 'Steel_Plate_Thickness' in data.columns:
        min_thickness = data['Steel_Plate_Thickness'].min()
        max_thickness = data['Steel_Plate_Thickness'].max()
        if max_thickness != min_thickness:
            data['Normalized_Steel_Thickness'] = (data['Steel_Plate_Thickness'] - min_thickness) / (max_thickness - min_thickness)
        else:
            data['Normalized_Steel_Thickness'] = 0
    else:
        data['Normalized_Steel_Thickness'] = np.nan

    if all(col in data.columns for col in ['X_Maximum', 'X_Minimum', 'Pixels_Areas']):
        data['X_Range*Pixels_Areas'] = (data['X_Maximum'] - data['X_Minimum']) * data['Pixels_Areas']
    else:
        data['X_Range*Pixels_Areas'] = np.nan

    return data

FEATURES_TO_DROP = ['Y_Minimum', 'Steel_Plate_Thickness', 'Sum_of_Luminosity', 'Edges_X_Index', 'SigmoidOfAreas', 'Luminosity_Index', 'TypeOfSteel_A300']

TARGET_FEATURES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
label_mapping = {i + 1: TARGET_FEATURES[i] for i in range(len(TARGET_FEATURES))}
label_mapping[0] = 'No_Defect'
sorted_labels = sorted(label_mapping.keys())
# Convert to strings to ensure they're hashable
display_labels = [str(label_mapping[key]) for key in sorted_labels]

ARTIFACTS_FILE = r"all_deployment_artifacts.joblib"

try:
    artifacts = joblib.load(ARTIFACTS_FILE)
    imported_model = artifacts['cat']
except FileNotFoundError:
    print(f"Error: Model file '{ARTIFACTS_FILE}' not found.")
    imported_model = None

app = Flask(__name__)

@app.route('/')
def home():
    return "Steel Plate Defect Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    if imported_model is None:
        return jsonify({'error': 'Model or scaler not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid input data format. Expected dictionary or list of dictionaries.'}), 400

        processed_df = feature_engineering(input_df.copy())
        processed_df = processed_df.drop(FEATURES_TO_DROP, axis=1, errors='ignore')

        try:
            train_df_for_cols = pd.read_csv("train.csv")
            train_df_for_cols.drop(columns=['id'] + TARGET_FEATURES, inplace=True, errors='ignore')
            train_df_for_cols = feature_engineering(train_df_for_cols)
            train_cols_order = train_df_for_cols.drop(FEATURES_TO_DROP, axis=1, errors='ignore').columns.tolist()
        except FileNotFoundError:
            return jsonify({'error': "Could not load training data to determine column order."}), 500
        except Exception as e:
            return jsonify({'error': f"Error processing training data for column order: {e}"}), 500

        try:
            processed_df = processed_df.reindex(columns=train_cols_order, fill_value=np.nan)
        except Exception as e:
            return jsonify({'error': f"Error reindexing input columns: {e}"}), 400

        predictions_proba = imported_model.predict_proba(processed_df)
        predictions_indices = imported_model.predict(processed_df)

        # Ensure predictions_indices are Python integers, not numpy integers
        predictions_indices = [int(idx) for idx in predictions_indices]
        predictions_labels = [str(label_mapping.get(idx, 'Unknown')) for idx in predictions_indices]

        results = []
        for i in range(len(input_df)):
            # Ensure all values are native Python types for JSON serialization
            probabilities_dict = {}
            for j in range(predictions_proba.shape[1]):
                key = str(display_labels[j])  # Ensure key is string
                value = float(predictions_proba[i][j])  # Ensure value is Python float
                probabilities_dict[key] = value
            
            sample_result = {
                'predicted_label': predictions_labels[i],
                'predicted_class_index': predictions_indices[i],
                'probabilities': probabilities_dict
            }
            results.append(sample_result)

        return jsonify(results)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        print(traceback.format_exc())  # This will help debug the exact error
        return jsonify({'error': str(e)}), 500

@app.route('/predict-one', methods=['POST'])
def predict1():
    if imported_model is None:
        return jsonify({'error': 'Model or scaler not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        else:
            return jsonify({'error': 'Invalid input data format. Expected a dictionary.'}), 400

        processed_df = feature_engineering(input_df.copy())
        processed_df = processed_df.drop(FEATURES_TO_DROP, axis=1, errors='ignore')

        try:
            train_cols_order = ['X_Minimum',
            'X_Maximum',
            'Y_Maximum',
            'Pixels_Areas',
            'X_Perimeter',
            'Y_Perimeter',
            'Minimum_of_Luminosity',
            'Maximum_of_Luminosity',
            'Length_of_Conveyer',
            'TypeOfSteel_A400',
            'Edges_Index',
            'Empty_Index',
            'Square_Index',
            'Outside_X_Index',
            'Edges_Y_Index',
            'Outside_Global_Index',
            'LogOfAreas',
            'Log_X_Index',
            'Log_Y_Index',
            'Orientation_Index',
            'Ratio_Length_Thickness',
            'Normalized_Steel_Thickness',
            'X_Range*Pixels_Areas']
        except FileNotFoundError:
            print()
            return jsonify({'error': "Could not load training data to determine column order."}), 500

        try:
            processed_df = processed_df.reindex(columns=train_cols_order, fill_value=np.nan)
        except Exception as e:
            return jsonify({'error': f"Error reindexing input columns: {e}"}), 400

        predictions_proba = imported_model.predict_proba(processed_df)
        predictions_indices = imported_model.predict(processed_df)

        # Convert to native Python types
        predicted_index = int(predictions_indices[0])
        predicted_label = str(label_mapping.get(predicted_index, 'Unknown'))

        # Build probabilities dictionary safely
        probabilities_dict = {}
        for j in range(predictions_proba.shape[1]):
            key = str(display_labels[j])  # Ensure key is string
            value = float(predictions_proba[0][j])  # Ensure value is Python float
            probabilities_dict[key] = value

        result = {
            'predicted_label': predicted_label,
            'predicted_class_index': predicted_index,
            'probabilities': probabilities_dict
        }

        return jsonify(result)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)