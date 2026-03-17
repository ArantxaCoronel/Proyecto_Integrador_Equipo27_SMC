import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# sklearn imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler

# ipywidgets for the HMI
import ipywidgets as widgets
from IPython.display import display, HTML

# --- Helper functions ---
def parse_and_average_percentage(value):
    if isinstance(value, str) and value.strip():
        try:
            parts = value.replace('%', '').split('/')
            numbers = []
            for p in parts:
                try:
                    num = float(p.strip())
                    numbers.append(num / 100)
                except ValueError:
                    continue
            if numbers:
                return np.mean(numbers)
            else:
                return np.nan
        except Exception:
            return np.nan
    elif pd.isna(value):
        return np.nan
    else:
        return value

def parse_and_average_number(value):
    if isinstance(value, str) and value.strip():
        try:
            parts = value.split('/')
            numbers = []
            for p in parts:
                try:
                    numbers.append(float(p.strip()))
                except ValueError:
                    continue
            if numbers:
                return np.mean(numbers)
            else:
                return np.nan
        except Exception:
            return np.nan
    elif pd.isna(value):
        return np.nan
    else:
        return value

# --- Configuration ---
target_columns = ['1st Flush_lbs', '2nd Flush_lbs', '3rd Flush_lbs']

# This dictionary is used to understand which models are chosen per target.
# The actual fitted models are loaded from 'fitted_best_models.pkl'.
best_models_config = {
    '1st Flush_lbs': {
        'Model_Name': 'VotingRegressor',
        'R2_Test': np.float64(0.6292673814807158),
        'RMSE_Test': np.float64(1291.7360242833538),
        'MAE_Test': np.float64(1013.4678302505083),
        'Hyperparameters': {
            'estimators': [
                ('decisiontree', {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 10}),
                ('randomforest', {'max_depth': 20, 'max_features': 1.0, 'n_estimators': 100})
            ],
            'weights': [
                np.float64(0.6674672895180437),
                np.float64(0.5910674734433877)
            ]
        },
        'Used_PCA': False
    },
    '2nd Flush_lbs': {
        'Model_Name': 'AdaBoostRegressor',
        'R2_Test': np.float64(0.2936465483462669),
        'RMSE_Test': np.float64(1003.8953792876476),
        'MAE_Test': np.float64(782.6941882270279),
        'Hyperparameters': {'learning_rate': 0.5, 'loss': 'linear', 'n_estimators': 50},
        'Used_PCA': False
    },
    '3rd Flush_lbs': {
        'Model_Name': 'RandomForestRegressor',
        'R2_Test': np.float64(0.35977649664884237),
        'RMSE_Test': np.float64(498.9731067656133),
        'MAE_Test': np.float64(386.3018207404298),
        'Hyperparameters': {'max_depth': 10, 'max_features': 0.8, 'n_estimators': 200},
        'Used_PCA': False
    }
}

# --- Load Fitted Models and Scaler ---
try:
    fitted_best_models = joblib.load('fitted_best_models.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Models and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler files not found. Make sure 'fitted_best_models.pkl' and 'scaler.pkl' are in the same directory.")
    exit() # Exit if essential files are missing

# --- Feature Names and Default Values ---
# These features correspond to X_train_pre_pca.columns and need to be in the exact order
feature_names = [
    'Compost Information_Spawn Rate', 'Compost Information_Supplement Rate',
    'Compost Information_Water Added', 'Farm Filling Information_Sqft Filled',
    'Farm Filling Information_Missing Sections', 'Farm Filling Information_Filled heights (cm)',
    'Farm Filling Information_Filled kg', 'Farm Filling Information_Fill rate kg/m²',
    'Watering_Water Applied (L)_1st Flush', 'Watering_Water Applied (L)_2nd Flush',
    'Watering_Water Applied (L)_3rd Flush', 'Watering_Water Applied (L)_Total (L)',
    'Farm_Filling_Information_Loading_Date_Year', 'Days_Between_Loading_and_Picking'
]

# Hardcoded median values from the original X dataframe (before log1p/scaling)
# These are used as default values for the HMI to represent typical inputs
hardcoded_medians_for_defaults = {
    'Compost Information_Spawn Rate': 15.015348245,
    'Compost Information_Supplement Rate': 0.0135,
    'Compost Information_Water Added': 5.140000000000001,
    'Farm Filling Information_Sqft Filled': 4647.0,
    'Farm Filling Information_Missing Sections': 2.0,
    'Farm Filling Information_Filled heights (cm)': 29.0,
    'Farm Filling Information_Filled kg': 38140.0,
    'Farm Filling Information_Fill rate kg/m²': 88.779087905,
    'Watering_Water Applied (L)_1st Flush': 22.0,
    'Watering_Water Applied (L)_2nd Flush': 14.5,
    'Watering_Water Applied (L)_3rd Flush': 9.0,
    'Watering_Water Applied (L)_Total (L)': 31.0,
    'Farm_Filling_Information_Loading_Date_Year': 2025.0,
    'Days_Between_Loading_and_Picking': 15.0
}

def get_default_values():
    return hardcoded_medians_for_defaults

# --- Define Input Widgets for the Features ---
input_widgets = {}
current_default_values = get_default_values()

for feature in feature_names:
    truncated_description = feature.replace('_', ' ').title()
    common_layout = widgets.Layout(width='500px')

    # Specific handling for features that originally had complex string formats or special types
    if feature in ['Compost Information_Supplement Rate', 'Compost Information_Water Added', 'Farm Filling Information_Filled heights (cm)']:
        input_widgets[feature] = widgets.Text(
            description=truncated_description,
            value=str(current_default_values.get(feature, '')),
            layout=common_layout
        )
    elif 'Year' in feature or 'Day' in feature or 'Week' in feature:
        input_widgets[feature] = widgets.IntText(
            description=truncated_description,
            value=int(current_default_values.get(feature, 0)),
            layout=common_layout
        )
    else: # Default for all other numerical features
        input_widgets[feature] = widgets.FloatText(
            description=truncated_description,
            value=float(current_default_values.get(feature, 0.0)),
            layout=common_layout
        )

# --- Define Preprocessing Function for User Input ---
def preprocess_input(input_values_dict):
    # Create a DataFrame from the input values to maintain column order and structure
    input_df = pd.DataFrame([input_values_dict], columns=feature_names)

    # Apply mixed-type parsing functions
    if 'Compost Information_Supplement Rate' in input_df.columns:
        input_df['Compost Information_Supplement Rate'] = input_df['Compost Information_Supplement Rate'].apply(parse_and_average_percentage)
    if 'Compost Information_Water Added' in input_df.columns:
        input_df['Compost Information_Water Added'] = input_df['Compost Information_Water Added'].apply(parse_and_average_number)
    if 'Farm Filling Information_Filled heights (cm)' in input_df.columns:
        input_df['Farm Filling Information_Filled heights (cm)'] = pd.to_numeric(input_df['Compost Information_Filled heights (cm)'], errors='coerce')

    # Impute NaNs for specific columns using hardcoded medians (pre-log1p values)
    for col in ['Compost Information_Supplement Rate', 'Compost Information_Water Added', 'Farm Filling Information_Filled heights (cm)']:
        if col in input_df.columns and input_df[col].isnull().any():
            input_df[col] = input_df[col].fillna(hardcoded_medians_for_defaults.get(col, 0.0))

    # Apply log1p transformation to all numerical columns.
    for col in feature_names:
        input_df[col] = np.log1p(input_df[col])
        # Fallback for log1p-introduced NaNs (if any, e.g. from negative inputs or invalid parsing results)
        if input_df[col].isnull().any():
            # Fill with log1p of the hardcoded median for that feature
            input_df[col] = input_df[col].fillna(np.log1p(hardcoded_medians_for_defaults.get(col, 0.0)))

    # Scale the input using the fitted scaler
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    return input_scaled_df

# --- Define the Prediction Function for the HMI ---
def predict_yield(**kwargs):
    input_values = {name: value for name, value in kwargs.items()}
    preprocessed_input_scaled_df = preprocess_input(input_values)

    results = []
    total_predicted_lbs = 0

    for target in target_columns:
        model = fitted_best_models[target] # Use the loaded fitted model directly
        predicted_lbs = model.predict(preprocessed_input_scaled_df)[0]
        results.append(f"<li><b>{target}</b>: {predicted_lbs:.2f} lbs</li>")
        total_predicted_lbs += predicted_lbs

    display(HTML(f"<h3>Prediction Results:</h3><ul>" + "".join(results) + f"<li><b>Total Predicted (all flushes)</b>: {total_predicted_lbs:.2f} lbs</li></ul>"))

# --- Create and Display the Interactive HMI ---
input_widgets_list = [input_widgets[feature] for feature in feature_names]

predict_button = widgets.Button(description="Predict Yield")
output_area = widgets.Output()

def on_button_click(b):
    with output_area:
        output_area.clear_output()
        current_input_values = {name: widget.value for name, widget in input_widgets.items()}
        predict_yield(**current_input_values)

predict_button.on_click(on_button_click)

print("Interactive HMI generated. Copy and save the following script as 'prediction_tool.py'.")
display(widgets.VBox(input_widgets_list + [predict_button, output_area]))