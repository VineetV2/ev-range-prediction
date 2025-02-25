import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tabulate import tabulate

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

def enhance_temperature_features(data):
    """
    Enhances temperature features to better capture battery physics and HVAC impacts.

    Real-world EVs experience significant range reductions in extreme temperatures
    due to battery chemistry limitations and increased energy use for cabin climate control.
    """
    # Original quadratic temperature feature
    data['temp_squared'] = data['temperature_celsius'] ** 2

    # Add categorical temperature features representing different operating regimes
    data['temp_below_freezing'] = np.where(data['temperature_celsius'] < 0, 1, 0)
    data['optimal_temp_range'] = np.where(
        (data['temperature_celsius'] >= 15) &
        (data['temperature_celsius'] <= 25),
        1, 0)
    data['extreme_heat'] = np.where(data['temperature_celsius'] > 35, 1, 0)

    # Temperature effects often worsen as battery depletes
    data['temp_energy_interaction'] = data['temperature_celsius'] * data['cumulative_energy_kwh']

    # Cold weather requires battery heating and cabin heating (scaled down)
    data['cold_energy_draw'] = np.where(
        data['temperature_celsius'] < 10,
        (10 - data['temperature_celsius']) * 0.02,  # Reduced from 0.03
        0
    )

    return data

def enhance_speed_features(data):
    """
    Enhances speed features to better model aerodynamic effects at different speeds.

    This implementation uses a more realistic approach to modeling how speed affects
    range, with a more gradual decline at highway speeds.
    """
    # Original squared speed feature (retained but with less influence)
    data['speed_squared'] = data['speed_kph'] ** 2 / 500  # Scaled to reduce impact

    # Dramatically reduced cubic term to avoid excessive penalties at high speeds
    data['speed_cubed'] = data['speed_kph'] ** 3 / 500000  # Much smaller impact

    # Speed efficiency zones with more nuanced efficiency curves
    data['low_speed'] = np.where(data['speed_kph'] < 30, 1, 0)
    data['optimal_speed'] = np.where(
        (data['speed_kph'] >= 30) & (data['speed_kph'] <= 70),
        1, 0)
    data['high_speed'] = np.where(
        (data['speed_kph'] > 70) & (data['speed_kph'] <= 100),
        1, 0)
    data['very_high_speed'] = np.where(data['speed_kph'] > 100, 1, 0)

    # Distance from peak efficiency speed (typically around 45-55 km/h for many EVs)
    data['peak_efficiency_speed'] = abs(data['speed_kph'] - 50) / 50  # Normalized

    # Speed has different effects in different driving conditions
    data['speed_highway_interaction'] = data['speed_kph'] * data['driving_highway'] / 100  # Scaled

    # Logarithmic speed feature (grows more slowly than linear or exponential)
    data['log_speed'] = np.log1p(data['speed_kph'])

    return data

def add_elevation_load_features(data, elevation_data=None, load_data=None):
    """
    Adds elevation and vehicle load features that affect energy consumption.

    Uphill driving significantly increases energy use, while downhill sections
    allow for regenerative braking. Vehicle weight directly impacts energy needed.
    """
    # If real elevation data is available, use it
    if elevation_data is not None:
        data['elevation_m'] = elevation_data
    else:
        # For synthetic data, create random elevation values
        data['elevation_m'] = np.random.normal(0, 100, size=len(data))

    # Calculate elevation gradient (rate of climbing/descending)
    if len(data) > 1:
        data['elevation_gradient'] = np.gradient(data['elevation_m'])
    else:
        data['elevation_gradient'] = 0

    # Separate uphill and downhill effects
    data['uphill_driving'] = np.where(data['elevation_gradient'] > 0,
                                     data['elevation_gradient'], 0)
    data['downhill_driving'] = np.where(data['elevation_gradient'] < 0,
                                       abs(data['elevation_gradient']), 0)

    # Add vehicle load features
    if load_data is not None:
        data['vehicle_load_kg'] = load_data
    else:
        # Simulate different load scenarios for synthetic data
        data['vehicle_load_kg'] = np.random.choice(
            [0, 75, 150, 225, 300],  # Different passenger/cargo combinations
            size=len(data)
        )

    # Calculate load efficiency impact (every 100kg reduces efficiency by approximately 2%)
    data['load_impact'] = data['vehicle_load_kg'] * 0.0002  # Reduced from 0.0003

    return data

def add_hvac_features(data):
    """
    Adds climate control features that impact energy consumption.

    HVAC systems can be a major energy drain, especially in extreme temperatures
    where heating or cooling demands are high.
    """
    # Estimate air conditioning usage based on temperature
    # A/C usage increases as temperature rises above 25°C
    data['ac_usage'] = np.where(
        data['temperature_celsius'] > 25,
        (data['temperature_celsius'] - 25) * 0.02,  # Kept at 0.02
        0
    )

    # Estimate heater usage as temperature drops below 15°C
    data['heater_usage'] = np.where(
        data['temperature_celsius'] < 15,
        (15 - data['temperature_celsius']) * 0.02,  # Reduced from 0.03
        0
    )

    # Combined HVAC energy impact
    data['hvac_energy_impact'] = data['ac_usage'] + data['heater_usage']

    return data

def calculate_speed_efficiency(speed):
    """
    Calculate how speed affects energy efficiency using a piecewise function.

    This function implements a more gradual decline in efficiency as speeds increase,
    better matching real-world EV behavior.

    Args:
        speed: Vehicle speed in km/h

    Returns:
        Efficiency factor (1.0 = 100% efficient, lower values = less efficient)
    """
    # Range of speeds where EVs are most efficient (usually 30-60 km/h)
    if speed < 30:
        # City driving, stop and go, but regenerative braking helps
        return 0.95 + (speed / 30) * 0.05  # 0.95-1.0 efficiency
    elif speed <= 60:
        # Most efficient range for most EVs
        return 1.0
    elif speed <= 100:
        # Highway driving - gradual efficiency decrease
        # Linear decrease from 1.0 at 60 km/h to 0.8 at 100 km/h
        return 1.0 - ((speed - 60) / 40) * 0.2
    else:
        # Very high speed - further efficiency decrease but with a floor
        # Logarithmic decrease to prevent unrealistic predictions
        excess_speed = speed - 100
        return 0.8 - 0.1 * (np.log1p(excess_speed) / np.log1p(30))

def generate_enhanced_training_data(n_samples=10000):
    """
    Generates realistic synthetic training data with enhanced feature relationships.

    This function creates a diverse dataset that captures complex relationships
    between driving conditions, environmental factors, and battery range while
    ensuring physically realistic predictions.
    """
    print(f"Generating {n_samples} synthetic training samples...")

    # Create base dataframe with core features
    data = pd.DataFrame({
        'speed_kph': np.random.uniform(0, 130, n_samples),
        'temperature_celsius': np.random.uniform(-20, 45, n_samples),
        'driving_type': np.random.choice(['city', 'highway', 'mixed'], n_samples),
        'cumulative_energy_kwh': np.random.uniform(0, 75, n_samples)
    })

    # One-hot encode driving type
    data['driving_city'] = np.where(data['driving_type'] == 'city', 1, 0)
    data['driving_highway'] = np.where(data['driving_type'] == 'highway', 1, 0)
    data['driving_mixed'] = np.where(data['driving_type'] == 'mixed', 1, 0)

    # Add all enhanced features
    print("Adding temperature features...")
    data = enhance_temperature_features(data)

    print("Adding speed features...")
    data = enhance_speed_features(data)

    print("Adding elevation and load features...")
    data = add_elevation_load_features(data)

    print("Adding HVAC features...")
    data = add_hvac_features(data)

    # Create target variable (range_km) with realistic physics-based relationships
    print("Calculating realistic range values...")

    # Start with base range (fully charged battery at optimal conditions)
    base_range = 400  # km
    battery_capacity = 75  # kWh

    # Calculate remaining energy percentage
    remaining_energy_pct = (1 - (data['cumulative_energy_kwh'] / battery_capacity))

    # Calculate efficiency factors using improved methods

    # Temperature effects: cold and hot temperatures reduce efficiency, but less severely
    temp_efficiency = np.ones(len(data))

    # Cold temperature effects (progressively worse as temperature drops)
    cold_mask = data['temperature_celsius'] < 15
    temp_efficiency[cold_mask] = 1.0 - 0.01 * (15 - data['temperature_celsius'][cold_mask])

    # Hot temperature effects (progressively worse as temperature rises)
    hot_mask = data['temperature_celsius'] > 25
    temp_efficiency[hot_mask] = 1.0 - 0.01 * (data['temperature_celsius'][hot_mask] - 25)

    # Ensure efficiency stays within reasonable bounds
    temp_efficiency = np.clip(temp_efficiency, 0.7, 1.0)

    # Speed effects using our piecewise function
    speed_efficiency = np.array([calculate_speed_efficiency(s) for s in data['speed_kph']])

    # Driving type effects: city driving benefits from regenerative braking
    driving_efficiency = 1.0 + (0.1 * data['driving_city']) - (0.05 * data['driving_highway'])

    # Other impacts that directly reduce range
    hvac_impact = data['hvac_energy_impact'] * 3  # Kept at 3
    elevation_impact = data['uphill_driving'] * 0.1 - data['downhill_driving'] * 0.05
    load_impact = data['load_impact'] * base_range

    # Calculate combined efficiency factor
    combined_efficiency = temp_efficiency * speed_efficiency * driving_efficiency

    # Softened impact of combined negative factors to prevent extreme penalties
    # When multiple negative factors are present
    extreme_conditions = ((data['temp_below_freezing'] == 1) &
                         (data['speed_kph'] > 90) &
                         (data['cumulative_energy_kwh'] > 40))
    combined_efficiency[extreme_conditions] *= 1.3  # Apply a boost to prevent unrealistic results

    # Calculate final range with all factors and softened combined effects
    data['range_km'] = (remaining_energy_pct * base_range * combined_efficiency -
                      hvac_impact - elevation_impact - load_impact)

    # Implement a reasonable minimum range for highway speeds
    # Real EVs don't suddenly drop to near-zero range at high speeds
    highway_speeds = data['speed_kph'] > 80
    min_range_factor = 0.2 * (1 - (data['cumulative_energy_kwh'] / battery_capacity))
    min_range = base_range * min_range_factor

    # Apply minimum range for highway speeds based on remaining energy
    data.loc[highway_speeds, 'range_km'] = np.maximum(
        data.loc[highway_speeds, 'range_km'],
        min_range[highway_speeds]
    )

    # Ensure range is never below 1 km in the training data
    data['range_km'] = data['range_km'].clip(lower=1)

    # Add random noise to simulate real-world variations
    data['range_km'] += np.random.normal(0, 5, n_samples)

    # Final clip to ensure positivity after adding noise
    data['range_km'] = data['range_km'].clip(lower=1)

    print("Training data generation complete!")
    return data

def optimize_model(X_train, y_train, cv_folds=5):
    """
    Performs feature selection and hyperparameter tuning to optimize the model.

    This two-step process first identifies the most important features, then
    finds the best model parameters for those features.
    """
    print("Starting model optimization...")

    # Initialize base model for feature selection
    base_model = GradientBoostingRegressor(random_state=42)

    # Perform feature selection
    print("Performing feature selection...")
    feature_selector = SelectFromModel(base_model, threshold='median')
    feature_selector.fit(X_train, y_train)
    X_train_selected = feature_selector.transform(X_train)

    # Get selected feature indices and names
    selected_indices = feature_selector.get_support()
    selected_feature_names = X_train.columns[selected_indices].tolist()

    print(f"Selected {len(selected_feature_names)} out of {X_train.shape[1]} features:")
    print(", ".join(selected_feature_names))

    # Define parameter grid for hyperparameter tuning
    print(f"Tuning hyperparameters with {cv_folds}-fold cross-validation...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train_selected, y_train)

    # Get best parameters
    best_params = grid_search.best_params_
    print("Best parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Train final model with best parameters
    print("Training final model with optimized parameters...")
    optimized_model = GradientBoostingRegressor(
        random_state=42,
        **best_params
    )

    optimized_model.fit(X_train_selected, y_train)

    return optimized_model, feature_selector

def evaluate_model(model, feature_selector, X_test, y_test, scaler):
    """
    Evaluates the model performance on test data and visualizes feature importance.

    This function computes standard metrics and creates visualizations to understand
    model performance and feature contributions.
    """
    print("Evaluating model performance...")

    # Apply feature selector to test data
    X_test_selected = feature_selector.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_selected)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance Metrics:")
    print(f"Mean Absolute Error: {mae:.2f} km")
    print(f"Root Mean Squared Error: {rmse:.2f} km")
    print(f"R² Score: {r2:.3f}")

    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Range (km)')
    plt.ylabel('Predicted Range (km)')
    plt.title('Predicted vs Actual Range')
    plt.tight_layout()
    plt.savefig('prediction_accuracy.png')
    print("Saved prediction accuracy plot to 'prediction_accuracy.png'")

    # Get selected feature names
    selected_indices = feature_selector.get_support()
    selected_features = X_test.columns[selected_indices].tolist()

    # Feature importance visualization
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Range Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved feature importance plot to 'feature_importance.png'")

    return mae, rmse, r2

def test_model_on_scenarios(model, scaler, feature_selector, feature_names):
    """
    Tests the enhanced model on specific scenarios to verify improvements.
    This function handles feature names properly to avoid warnings.
    """
    print("\nTesting enhanced model on critical scenarios...")

    # Create test scenarios
    test_scenarios = [
        # Temperature tests
        {"name": "Very Cold (-20°C)", "speed": 60, "temp": -20, "type": "mixed", "energy": 10},
        {"name": "Cold (0°C)", "speed": 60, "temp": 0, "type": "mixed", "energy": 10},
        {"name": "Optimal (20°C)", "speed": 60, "temp": 20, "type": "mixed", "energy": 10},
        {"name": "Hot (35°C)", "speed": 60, "temp": 35, "type": "mixed", "energy": 10},

        # Speed tests
        {"name": "Low Speed (20 km/h)", "speed": 20, "temp": 20, "type": "highway", "energy": 10},
        {"name": "Medium Speed (60 km/h)", "speed": 60, "temp": 20, "type": "highway", "energy": 10},
        {"name": "High Speed (100 km/h)", "speed": 100, "temp": 20, "type": "highway", "energy": 10},
        {"name": "Very High Speed (130 km/h)", "speed": 130, "temp": 20, "type": "highway", "energy": 10}
    ]

    results = []

    # Function to prepare a single input row with correct feature handling
    def prepare_input_row(scenario):
        # Create dataframe with base features
        input_data = pd.DataFrame({
            'speed_kph': [scenario["speed"]],
            'temperature_celsius': [scenario["temp"]],
            'cumulative_energy_kwh': [scenario["energy"]],
            'driving_city': [1 if scenario["type"] == "city" else 0],
            'driving_highway': [1 if scenario["type"] == "highway" else 0],
            'driving_mixed': [1 if scenario["type"] == "mixed" else 0]
        })

        # Add all enhanced features
        input_data = enhance_temperature_features(input_data)
        input_data = enhance_speed_features(input_data)
        input_data = add_hvac_features(input_data)

        # For elevation and load features, use reasonable defaults
        input_data['elevation_m'] = [0]
        input_data['elevation_gradient'] = [0]
        input_data['uphill_driving'] = [0]
        input_data['downhill_driving'] = [0]
        input_data['vehicle_load_kg'] = [0]
        input_data['load_impact'] = [0]

        # Create a dataframe with all required features in the correct order
        full_input = pd.DataFrame(index=[0], columns=feature_names)
        for col in feature_names:
            if col in input_data.columns:
                full_input[col] = input_data[col].values
            else:
                full_input[col] = 0

        return full_input

    for scenario in test_scenarios:
        # Prepare input with correct feature ordering
        input_data = prepare_input_row(scenario)

        # Scale the data
        input_scaled = scaler.transform(input_data)

        # Create a DataFrame with the same column names for feature selection
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

        # Select features using the trained feature selector
        input_selected = feature_selector.transform(input_scaled_df)

        # Make prediction
        prediction = model.predict(input_selected)[0]

        # Ensure non-negative prediction
        prediction = max(1.0, prediction)

        # Store result
        results.append({
            'scenario': scenario["name"],
            'speed_kph': scenario["speed"],
            'temperature_celsius': scenario["temp"],
            'driving_type': scenario["type"],
            'energy_used_kwh': scenario["energy"],
            'predicted_range_km': prediction
        })

        print(f"Scenario: {scenario['name']}, Predicted Range: {prediction:.1f} km")

    # Convert to DataFrame for visualization
    results_df = pd.DataFrame(results)

    # Visualize temperature effect
    plt.figure(figsize=(12, 6))
    temp_results = results_df[results_df['speed_kph'] == 60].copy()
    sns.barplot(data=temp_results, x='temperature_celsius', y='predicted_range_km')
    plt.title('Effect of Temperature on Predicted Range')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Predicted Range (km)')
    plt.tight_layout()
    plt.savefig('temperature_effect.png')
    print("Saved temperature effect plot to 'temperature_effect.png'")

    # Visualize speed effect
    plt.figure(figsize=(12, 6))
    speed_results = results_df[results_df['temperature_celsius'] == 20].copy()
    sns.barplot(data=speed_results, x='speed_kph', y='predicted_range_km')
    plt.title('Effect of Speed on Predicted Range')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Predicted Range (km)')
    plt.tight_layout()
    plt.savefig('speed_effect.png')
    print("Saved speed effect plot to 'speed_effect.png'")

    return results_df

def predict_range(speed, temperature, driving_type, energy_used,
                 elevation=0, vehicle_load=0, model_dir='model'):
    """
    Makes a range prediction using the enhanced model with proper feature handling.

    Parameters:
    -----------
    speed : float
        Current vehicle speed in km/h
    temperature : float
        Ambient temperature in Celsius
    driving_type : str
        Type of driving ('city', 'highway', or 'mixed')
    energy_used : float
        Energy consumed from battery in kWh
    elevation : float, optional
        Current elevation gradient (default 0)
    vehicle_load : float, optional
        Additional vehicle load in kg (default 0)
    model_dir : str, optional
        Directory where model files are saved

    Returns:
    --------
    float
        Predicted remaining range in kilometers
    """
    # Load model components
    model_path = os.path.join(model_dir, 'enhanced_ev_model.joblib')
    scaler_path = os.path.join(model_dir, 'enhanced_ev_scaler.joblib')
    feature_selector_path = os.path.join(model_dir, 'enhanced_ev_feature_selector.joblib')
    feature_names_path = os.path.join(model_dir, 'feature_names.joblib')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_selector = joblib.load(feature_selector_path)
    feature_names = joblib.load(feature_names_path)

    # Create base input data
    input_data = pd.DataFrame({
        'speed_kph': [speed],
        'temperature_celsius': [temperature],
        'cumulative_energy_kwh': [energy_used],
        'driving_city': [1 if driving_type == 'city' else 0],
        'driving_highway': [1 if driving_type == 'highway' else 0],
        'driving_mixed': [1 if driving_type == 'mixed' else 0]
    })

    # Add enhanced features exactly as during training
    input_data = enhance_temperature_features(input_data)
    input_data = enhance_speed_features(input_data)
    input_data = add_hvac_features(input_data)

    # Add elevation and load features
    input_data['elevation_m'] = [0]
    input_data['elevation_gradient'] = [elevation]
    input_data['uphill_driving'] = [max(0, elevation)]
    input_data['downhill_driving'] = [max(0, -elevation)]
    input_data['vehicle_load_kg'] = [vehicle_load]
    input_data['load_impact'] = [vehicle_load * 0.0002]  # Match training calculation

    # Create a dataframe with all required features in the correct order
    full_input = pd.DataFrame(index=[0], columns=feature_names)
    for col in feature_names:
        if col in input_data.columns:
            full_input[col] = input_data[col].values
        else:
            full_input[col] = 0

    # Scale the data
    input_scaled = scaler.transform(full_input)

    # Create a DataFrame with the same column names for feature selection
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    # Apply feature selection
    input_selected = feature_selector.transform(input_scaled_df)

    # Make prediction
    prediction = model.predict(input_selected)[0]

    # Apply a minimum reasonable range for highway speeds
    if speed > 80:
        # Calculate reasonable minimum range based on energy used
        battery_capacity = 75  # kWh
        remaining_pct = 1 - (energy_used / battery_capacity)
        min_range = max(50 * remaining_pct, 1)  # At least 50km * remaining battery percentage
        prediction = max(prediction, min_range)

    # Ensure non-negative range
    return max(1.0, prediction)

def demo_model_predictions():
    """
    Demonstrates the model with realistic scenarios.
    """
    print("\n" + "=" * 80)
    print("EV RANGE PREDICTION DEMONSTRATION")
    print("=" * 80)

    # Define scenarios to test
    scenarios = [
        {
            "name": "Optimal Conditions",
            "speed": 60,
            "temperature": 20,
            "driving_type": "mixed",
            "energy_used": 5,
            "elevation": 0,
            "vehicle_load": 0
        },
        {
            "name": "Winter Highway",
            "speed": 100,
            "temperature": -10,
            "driving_type": "highway",
            "energy_used": 15,
            "elevation": 0,
            "vehicle_load": 0
        },
        {
            "name": "Summer City with AC",
            "speed": 30,
            "temperature": 35,
            "driving_type": "city",
            "energy_used": 10,
            "elevation": 0,
            "vehicle_load": 0
        },
        {
            "name": "Family Road Trip",
            "speed": 80,
            "temperature": 22,
            "driving_type": "highway",
            "energy_used": 20,
            "elevation": 0,
            "vehicle_load": 250
        },
        {
            "name": "Mountain Driving",
            "speed": 50,
            "temperature": 15,
            "driving_type": "mixed",
            "energy_used": 30,
            "elevation": 5,
            "vehicle_load": 75
        }
    ]

    # Create a table to store results
    results = []

    # Run predictions for each scenario
    for scenario in scenarios:
        prediction = predict_range(
            speed=scenario["speed"],
            temperature=scenario["temperature"],
            driving_type=scenario["driving_type"],
            energy_used=scenario["energy_used"],
            elevation=scenario.get("elevation", 0),
            vehicle_load=scenario.get("vehicle_load", 0)
        )

        # Print detailed scenario information
        print(f"\nScenario: {scenario['name']}")
        print(f"  Speed: {scenario['speed']} km/h")
        print(f"  Temperature: {scenario['temperature']}°C")
        print(f"  Driving type: {scenario['driving_type']}")
        print(f"  Energy used: {scenario['energy_used']} kWh")
        print(f"  Elevation gradient: {scenario.get('elevation', 0)}")
        print(f"  Vehicle load: {scenario.get('vehicle_load', 0)} kg")
        print(f"Predicted Range: {prediction:.1f} km")

        # Store the result
        scenario_copy = scenario.copy()
        scenario_copy["predicted_range"] = prediction
        results.append(scenario_copy)

    # Create a comparison table and save it
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[["name", "speed", "temperature", "driving_type",
                                 "energy_used", "elevation", "vehicle_load", "predicted_range"]]
    comparison_df.columns = ["Scenario", "Speed (km/h)", "Temperature (°C)", "Driving Type",
                           "Energy Used (kWh)", "Elevation (%)", "Vehicle Load (kg)", "Predicted Range (km)"]

    # Save comparison table to file
    with open("../scenario_comparisons.txt", "w") as f:
        f.write(tabulate(comparison_df, headers="keys", tablefmt="grid"))

    print("\nScenario comparison table saved to 'scenario_comparisons.txt'")
    return results

def train_enhanced_ev_model():
    """
    Complete pipeline to train the enhanced EV range prediction model.

    This function handles the entire process from data generation to model
    training, evaluation, and testing.
    """
    print("=" * 80)
    print("ENHANCED EV RANGE PREDICTION MODEL TRAINING")
    print("=" * 80)

    # Generate enhanced training data
    data = generate_enhanced_training_data(n_samples=15000)

    # Define features and target
    print("\nPreparing training and test datasets...")
    feature_cols = [col for col in data.columns if col != 'range_km' and col != 'driving_type']
    X = data[feature_cols]
    y = data['range_km']

    # Save feature names for consistent ordering
    feature_names = X.columns.tolist()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for feature selection with column names preserved
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Optimize model through feature selection and hyperparameter tuning
    optimized_model, feature_selector = optimize_model(X_train_df, y_train)

    # Evaluate model performance
    mae, rmse, r2 = evaluate_model(
        optimized_model, feature_selector, X_test_df, y_test, scaler
    )

    # Test model on specific scenarios
    test_results = test_model_on_scenarios(optimized_model, scaler, feature_selector, feature_names)

    # Create model output directory if it doesn't exist
    if not os.path.exists('../model'):
        os.makedirs('../model')

    # Save the enhanced model components
    print("\nSaving model components...")
    joblib.dump(optimized_model, '../model/enhanced_ev_model.joblib')
    joblib.dump(scaler, '../model/enhanced_ev_scaler.joblib')
    joblib.dump(feature_selector, '../model/enhanced_ev_feature_selector.joblib')
    joblib.dump(feature_names, '../model/feature_names.joblib')

    print("\nModel training and evaluation complete!")
    print(f"Final model performance: MAE={mae:.2f} km, RMSE={rmse:.2f} km, R²={r2:.3f}")
    print("Model files saved in 'model' directory:")
    print("  - enhanced_ev_model.joblib")
    print("  - enhanced_ev_scaler.joblib")
    print("  - enhanced_ev_feature_selector.joblib")
    print("  - feature_names.joblib")

    # Return model components for further use if needed
    return optimized_model, scaler, feature_selector, feature_names, test_results

if __name__ == "__main__":
    # Train and evaluate the enhanced model
    model, scaler, feature_selector, feature_names, test_results = train_enhanced_ev_model()

    # Run demonstration predictions
    demo_model_predictions()