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
            return np.nan
        except Exception:
            return np.nan
    elif pd.isna(value):
        return np.nan
    else:
        return value


# --- Configuration ---
target_columns = ['1st Flush_lbs', '2nd Flush_lbs', '3rd Flush_lbs']

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
    print('Models and scaler loaded successfully.')
except FileNotFoundError:
    print("Error: Model or scaler files not found. Make sure 'fitted_best_models.pkl' and 'scaler.pkl' are in the same directory.")
    raise SystemExit


# --- Feature Names and Default Values ---
feature_names = [
    'Compost Information_Spawn Rate', 'Compost Information_Supplement Rate',
    'Compost Information_Water Added', 'Farm Filling Information_Sqft Filled',
    'Farm Filling Information_Missing Sections', 'Farm Filling Information_Filled heights (cm)',
    'Farm Filling Information_Filled kg', 'Farm Filling Information_Fill rate kg/m²',
    'Watering_Water Applied (L)_1st Flush', 'Watering_Water Applied (L)_2nd Flush',
    'Watering_Water Applied (L)_3rd Flush', 'Watering_Water Applied (L)_Total (L)',
    'Farm_Filling_Information_Loading_Date_Year', 'Days_Between_Loading_and_Picking'
]

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
watering_total_display = None
current_default_values = get_default_values()

container_width = '100%'
container_max_width = '1080px'
card_width = '100%'
card_height = '228px'
label_width = '260px'
field_width = '100px'
row_gap = '10px'
action_width = '100%'

feature_groups = {
    'Compost Information': [
        'Compost Information_Spawn Rate',
        'Compost Information_Supplement Rate',
        'Compost Information_Water Added'
    ],
    'Farm Filling Information': [
        'Farm Filling Information_Sqft Filled',
        'Farm Filling Information_Missing Sections',
        'Farm Filling Information_Filled heights (cm)',
        'Farm Filling Information_Filled kg',
        'Farm Filling Information_Fill rate kg/m²'
    ],
    'Watering': [
        'Watering_Water Applied (L)_1st Flush',
        'Watering_Water Applied (L)_2nd Flush',
        'Watering_Water Applied (L)_3rd Flush'
    ],
    'Date and Timing': [
        'Farm_Filling_Information_Loading_Date_Year',
        'Days_Between_Loading_and_Picking'
    ]
}


def format_feature_label(feature_name):
    return feature_name.replace('_', ' ')


def build_input_widget(feature):
    common_layout = widgets.Layout(width=field_width)

    if 'Year' in feature or 'Day' in feature or 'Week' in feature:
        widget = widgets.IntText(
            value=int(current_default_values.get(feature, 0)),
            layout=common_layout,
            style={'description_width': 'initial'}
        )
    else:
        widget = widgets.FloatText(
            value=float(current_default_values.get(feature, 0.0)),
            layout=common_layout,
            style={'description_width': 'initial'}
        )

    input_widgets[feature] = widget

    label_widget = widgets.HTML(
        value=(
            "<div style='font-size:13.5px; line-height:1.35; white-space:normal; "
            "overflow-wrap:anywhere;'><b>" + format_feature_label(feature) + "</b></div>"
        ),
        layout=widgets.Layout(width=label_width, min_width='0')
    )

    row = widgets.GridBox(
        [label_widget, widget],
        layout=widgets.Layout(
            width='100%',
            grid_template_columns=f'minmax(0, {label_width}) {field_width}',
            column_gap=row_gap,
            align_items='center',
            margin='0 0 10px 0',
            overflow='hidden'
        )
    )
    return row


def build_section(title, features):
    global watering_total_display

    section_header = widgets.HTML(
        value=f"""
        <div style='margin:0 0 12px 0;'>
            <h3 style='margin:0; font-size:17px;'>{title}</h3>
        </div>
        """
    )

    section_rows = [build_input_widget(feature) for feature in features]

    if title == 'Watering':
        watering_total_display = widgets.FloatText(
            value=45.50,
            disabled=True,
            layout=widgets.Layout(width=field_width),
            style={'description_width': 'initial'}
        )

        total_label_widget = widgets.HTML(
            value="<div style='font-size:13.5px; line-height:1.35; white-space:normal; overflow-wrap:anywhere;'><b>Watering Water Applied (L) Total (L)</b></div>",
            layout=widgets.Layout(width=label_width, min_width='0')
        )

        total_row = widgets.GridBox(
            [total_label_widget, watering_total_display],
            layout=widgets.Layout(
                width='100%',
                grid_template_columns=f'minmax(0, {label_width}) {field_width}',
                column_gap=row_gap,
                align_items='center',
                margin='0 0 10px 0',
                overflow='hidden'
            )
        )
        section_rows.append(total_row)

    section_box = widgets.VBox(
        [section_header] + section_rows,
        layout=widgets.Layout(
            width=card_width,
            min_height=card_height,
            padding='14px 16px 10px 16px',
            margin='0',
            border='1px solid #d9d9d9',
            overflow='hidden'
        )
    )
    return section_box


section_order = [
    'Watering',
    'Farm Filling Information',
    'Compost Information',
    'Date and Timing'
]

section_boxes = [build_section(group_name, feature_groups[group_name]) for group_name in section_order]

form_grid = widgets.GridBox(
    section_boxes,
    layout=widgets.Layout(
        width=container_width,
        max_width=container_max_width,
        margin='0 auto',
        grid_template_columns='repeat(2, minmax(0, 1fr))',
        grid_gap='14px 18px',
        align_items='stretch',
        justify_content='center'
    )
)


def calculate_total_water():
    return (
        float(input_widgets['Watering_Water Applied (L)_1st Flush'].value)
        + float(input_widgets['Watering_Water Applied (L)_2nd Flush'].value)
        + float(input_widgets['Watering_Water Applied (L)_3rd Flush'].value)
    )


def update_total_water_display(change=None):
    total_water = calculate_total_water()
    if watering_total_display is not None:
        watering_total_display.value = round(total_water, 2)


# --- Define Preprocessing Function for User Input ---
def preprocess_input(input_values_dict):
    input_df = pd.DataFrame([input_values_dict], columns=feature_names)

    if 'Compost Information_Supplement Rate' in input_df.columns:
        input_df['Compost Information_Supplement Rate'] = input_df['Compost Information_Supplement Rate'].apply(parse_and_average_percentage)
    if 'Compost Information_Water Added' in input_df.columns:
        input_df['Compost Information_Water Added'] = input_df['Compost Information_Water Added'].apply(parse_and_average_number)
    if 'Farm Filling Information_Filled heights (cm)' in input_df.columns:
        input_df['Farm Filling Information_Filled heights (cm)'] = pd.to_numeric(
            input_df['Farm Filling Information_Filled heights (cm)'], errors='coerce'
        )

    for col in [
        'Compost Information_Supplement Rate',
        'Compost Information_Water Added',
        'Farm Filling Information_Filled heights (cm)'
    ]:
        if col in input_df.columns and input_df[col].isnull().any():
            input_df[col] = input_df[col].fillna(hardcoded_medians_for_defaults.get(col, 0.0))

    for col in feature_names:
        input_df[col] = np.log1p(input_df[col])
        if input_df[col].isnull().any():
            input_df[col] = input_df[col].fillna(np.log1p(hardcoded_medians_for_defaults.get(col, 0.0)))

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
        model = fitted_best_models[target]
        predicted_lbs = model.predict(preprocessed_input_scaled_df)[0]
        results.append(f'<li><b>{target}</b>: {predicted_lbs:.2f} lbs</li>')
        total_predicted_lbs += predicted_lbs

    display(HTML(
        """
        <div style='margin-top:12px; padding:14px 18px; border:1px solid #d9d9d9; border-radius:10px;'>
            <h3 style='margin:0 0 10px 0;'>Prediction Results</h3>
            <ul style='margin:0 0 0 18px; padding:0;'>
        """ + ''.join(results) + f'<li><b>Total Predicted (all flushes)</b>: {total_predicted_lbs:.2f} lbs</li></ul></div>'
    ))


# --- Create and Display the Interactive HMI ---
predict_button = widgets.Button(
    description='Predict Yield',
    button_style='primary',
    layout=widgets.Layout(width='170px', height='40px', margin='0')
)
output_area = widgets.Output(layout=widgets.Layout(width=action_width))

form_title = widgets.HTML(
    value="""
    <div style='margin-bottom:14px;'>
        <h2 style='margin:0 0 6px 0;'>Yield Prediction Tool</h2>
        <p style='margin:0; color:#666;'>Enter the input values below to estimate mushroom yield by flush.</p>
    </div>
    """,
    layout=widgets.Layout(width=container_width, max_width=container_max_width)
)

button_row = widgets.HBox(
    [predict_button],
    layout=widgets.Layout(width=container_width, max_width=container_max_width, margin='10px auto 0 auto', justify_content='flex-start')
)


def on_button_click(b):
    with output_area:
        output_area.clear_output()
        current_input_values = {name: widget.value for name, widget in input_widgets.items()}
        current_input_values['Watering_Water Applied (L)_Total (L)'] = calculate_total_water()
        predict_yield(**current_input_values)


for feature_name in [
    'Watering_Water Applied (L)_1st Flush',
    'Watering_Water Applied (L)_2nd Flush',
    'Watering_Water Applied (L)_3rd Flush'
]:
    input_widgets[feature_name].observe(update_total_water_display, names='value')

update_total_water_display()
predict_button.on_click(on_button_click)

main_form = widgets.VBox(
    [form_title, form_grid, button_row, output_area],
    layout=widgets.Layout(width=container_width, max_width=container_max_width, padding='10px 8px 10px 8px', margin='0 auto')
)

print("Interactive HMI generated. Copy and save the following script as 'PredictionTool_4.py'.")
display(main_form)