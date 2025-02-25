"""
EV Range Prediction Model - Testing and Visualization Tool

This script provides tools to test the enhanced EV range prediction model
with various parameters and visualize the results to understand how different
factors affect range predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D



class EVModelTester:
    def __init__(self, model_dir='model'):
        """
        Initializes the tester by loading the trained model components.
        """
        # Load model components
        self.model_path = os.path.join(model_dir, 'enhanced_ev_model.joblib')
        self.scaler_path = os.path.join(model_dir, 'enhanced_ev_scaler.joblib')
        self.feature_selector_path = os.path.join(model_dir, 'enhanced_ev_feature_selector.joblib')
        self.feature_names_path = os.path.join(model_dir, 'feature_names.joblib')

        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.feature_selector = joblib.load(self.feature_selector_path)
            self.feature_names = joblib.load(self.feature_names_path)
            print("Model components loaded successfully!")
        except FileNotFoundError:
            print("Error: Model files not found. Please ensure the model has been trained.")
            raise

        # Set up styling for visualizations
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_context("talk")

    def enhance_temperature_features(self, data):
        """Enhances temperature features for the input data."""
        data['temp_squared'] = data['temperature_celsius'] ** 2
        data['temp_below_freezing'] = np.where(data['temperature_celsius'] < 0, 1, 0)
        data['optimal_temp_range'] = np.where(
            (data['temperature_celsius'] >= 15) &
            (data['temperature_celsius'] <= 25),
            1, 0)
        data['extreme_heat'] = np.where(data['temperature_celsius'] > 35, 1, 0)
        data['temp_energy_interaction'] = data['temperature_celsius'] * data['cumulative_energy_kwh']
        data['cold_energy_draw'] = np.where(
            data['temperature_celsius'] < 10,
            (10 - data['temperature_celsius']) * 0.02,
            0
        )
        return data

    def enhance_speed_features(self, data):
        """Enhances speed features for the input data."""
        data['speed_squared'] = data['speed_kph'] ** 2 / 500
        data['speed_cubed'] = data['speed_kph'] ** 3 / 500000
        data['low_speed'] = np.where(data['speed_kph'] < 30, 1, 0)
        data['optimal_speed'] = np.where(
            (data['speed_kph'] >= 30) & (data['speed_kph'] <= 70),
            1, 0)
        data['high_speed'] = np.where(
            (data['speed_kph'] > 70) & (data['speed_kph'] <= 100),
            1, 0)
        data['very_high_speed'] = np.where(data['speed_kph'] > 100, 1, 0)
        data['peak_efficiency_speed'] = abs(data['speed_kph'] - 50) / 50
        data['speed_highway_interaction'] = data['speed_kph'] * data['driving_highway'] / 100
        data['log_speed'] = np.log1p(data['speed_kph'])
        return data

    def add_hvac_features(self, data):
        """Adds HVAC features for the input data."""
        data['ac_usage'] = np.where(
            data['temperature_celsius'] > 25,
            (data['temperature_celsius'] - 25) * 0.02,
            0
        )
        data['heater_usage'] = np.where(
            data['temperature_celsius'] < 15,
            (15 - data['temperature_celsius']) * 0.02,
            0
        )
        data['hvac_energy_impact'] = data['ac_usage'] + data['heater_usage']
        return data

    def predict_range(self, speed, temperature, driving_type, energy_used,
                      elevation=0, vehicle_load=0):
        """
        Predicts the range using the enhanced model.
        """
        # Create base input data
        input_data = pd.DataFrame({
            'speed_kph': [speed],
            'temperature_celsius': [temperature],
            'cumulative_energy_kwh': [energy_used],
            'driving_city': [1 if driving_type == 'city' else 0],
            'driving_highway': [1 if driving_type == 'highway' else 0],
            'driving_mixed': [1 if driving_type == 'mixed' else 0]
        })

        # Add enhanced features
        input_data = self.enhance_temperature_features(input_data)
        input_data = self.enhance_speed_features(input_data)
        input_data = self.add_hvac_features(input_data)

        # Add elevation and load features
        input_data['elevation_m'] = [0]
        input_data['elevation_gradient'] = [elevation]
        input_data['uphill_driving'] = [max(0, elevation)]
        input_data['downhill_driving'] = [max(0, -elevation)]
        input_data['vehicle_load_kg'] = [vehicle_load]
        input_data['load_impact'] = [vehicle_load * 0.0002]

        # Create a dataframe with all required features in the correct order
        full_input = pd.DataFrame(index=[0], columns=self.feature_names)
        for col in self.feature_names:
            if col in input_data.columns:
                full_input[col] = input_data[col].values
            else:
                full_input[col] = 0

        # Scale the data
        input_scaled = self.scaler.transform(full_input)

        # Create a DataFrame with the same column names for feature selection
        input_scaled_df = pd.DataFrame(input_scaled, columns=self.feature_names)

        # Apply feature selection
        input_selected = self.feature_selector.transform(input_scaled_df)

        # Make prediction
        prediction = self.model.predict(input_selected)[0]

        # Apply a minimum reasonable range for highway speeds
        if speed > 80:
            # Calculate reasonable minimum range based on energy used
            battery_capacity = 75  # kWh
            remaining_pct = 1 - (energy_used / battery_capacity)
            min_range = max(50 * remaining_pct, 1)
            prediction = max(prediction, min_range)

        # Ensure non-negative range
        return max(1.0, prediction)

    def visualize_speed_impact(self, temperatures=[20], driving_type='mixed', energy_used=10,
                               speed_range=(10, 130), load=0, elevation=0):
        """
        Visualizes how speed affects predicted range at different temperatures.
        """
        speeds = np.linspace(speed_range[0], speed_range[1], 25)

        plt.figure(figsize=(12, 8))

        for temp in temperatures:
            ranges = []
            for speed in speeds:
                range_prediction = self.predict_range(
                    speed=speed,
                    temperature=temp,
                    driving_type=driving_type,
                    energy_used=energy_used,
                    elevation=elevation,
                    vehicle_load=load
                )
                ranges.append(range_prediction)

            plt.plot(speeds, ranges, 'o-', linewidth=2, label=f'Temperature: {temp}°C')

        plt.title(f'Impact of Speed on Predicted Range\n({driving_type.capitalize()} driving, {energy_used} kWh used)',
                  fontsize=16)
        plt.xlabel('Speed (km/h)', fontsize=14)
        plt.ylabel('Predicted Range (km)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('speed_impact_visualization.png', dpi=300)
        plt.show()

        return speeds, ranges

    def visualize_temperature_impact(self, speeds=[60], driving_type='mixed', energy_used=10,
                                     temp_range=(-20, 40), load=0, elevation=0):
        """
        Visualizes how temperature affects predicted range at different speeds.
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], 25)

        plt.figure(figsize=(12, 8))

        for speed in speeds:
            ranges = []
            for temp in temperatures:
                range_prediction = self.predict_range(
                    speed=speed,
                    temperature=temp,
                    driving_type=driving_type,
                    energy_used=energy_used,
                    elevation=elevation,
                    vehicle_load=load
                )
                ranges.append(range_prediction)

            plt.plot(temperatures, ranges, 'o-', linewidth=2, label=f'Speed: {speed} km/h')

        plt.title(
            f'Impact of Temperature on Predicted Range\n({driving_type.capitalize()} driving, {energy_used} kWh used)',
            fontsize=16)
        plt.xlabel('Temperature (°C)', fontsize=14)
        plt.ylabel('Predicted Range (km)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('temperature_impact_visualization.png', dpi=300)
        plt.show()

        return temperatures, ranges

    def visualize_battery_impact(self, speeds=[60], temperatures=[20],
                                 driving_type='mixed', energy_range=(0, 70), load=0, elevation=0):
        """
        Visualizes how battery consumption affects predicted range.
        """
        energy_values = np.linspace(energy_range[0], energy_range[1], 25)

        plt.figure(figsize=(12, 8))

        for speed in speeds:
            for temp in temperatures:
                ranges = []
                for energy in energy_values:
                    range_prediction = self.predict_range(
                        speed=speed,
                        temperature=temp,
                        driving_type=driving_type,
                        energy_used=energy,
                        elevation=elevation,
                        vehicle_load=load
                    )
                    ranges.append(range_prediction)

                plt.plot(energy_values, ranges, 'o-', linewidth=2,
                         label=f'Speed: {speed} km/h, Temp: {temp}°C')

        plt.title(f'Impact of Energy Consumption on Predicted Range\n({driving_type.capitalize()} driving)',
                  fontsize=16)
        plt.xlabel('Energy Used (kWh)', fontsize=14)
        plt.ylabel('Predicted Range (km)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('battery_impact_visualization.png', dpi=300)
        plt.show()

        return energy_values, ranges

    def create_3d_visualization(self, factor1='speed', factor2='temperature', energy_used=10,
                                driving_type='mixed', load=0, elevation=0):
        """
        Creates a 3D visualization showing how two factors interact to affect range.

        Parameters:
            factor1: 'speed' or 'temperature' or 'energy'
            factor2: 'speed' or 'temperature' or 'energy'
        """
        if factor1 == factor2:
            print("Error: The two factors must be different")
            return

        # Define ranges for each factor
        speed_vals = np.linspace(10, 130, 15)
        temp_vals = np.linspace(-20, 40, 15)
        energy_vals = np.linspace(0, 70, 15)

        # Determine which values to use for each axis
        if factor1 == 'speed':
            x_vals = speed_vals
            x_label = 'Speed (km/h)'
        elif factor1 == 'temperature':
            x_vals = temp_vals
            x_label = 'Temperature (°C)'
        else:  # energy
            x_vals = energy_vals
            x_label = 'Energy Used (kWh)'

        if factor2 == 'speed':
            y_vals = speed_vals
            y_label = 'Speed (km/h)'
        elif factor2 == 'temperature':
            y_vals = temp_vals
            y_label = 'Temperature (°C)'
        else:  # energy
            y_vals = energy_vals
            y_label = 'Energy Used (kWh)'

        # Create meshgrid
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)

        # Calculate range predictions for each combination
        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                # Set the parameters based on which factors we're using
                if factor1 == 'speed':
                    speed = x_val
                elif factor1 == 'temperature':
                    temperature = x_val
                else:  # energy
                    energy_used_val = x_val

                if factor2 == 'speed':
                    speed = y_val
                elif factor2 == 'temperature':
                    temperature = y_val
                else:  # energy
                    energy_used_val = y_val

                # Use default values for any parameter not specified by factor1 or factor2
                speed_to_use = speed if 'speed' in [factor1, factor2] else 60
                temp_to_use = temperature if 'temperature' in [factor1, factor2] else 20
                energy_to_use = energy_used_val if 'energy' in [factor1, factor2] else energy_used

                # Get prediction
                Z[i, j] = self.predict_range(
                    speed=speed_to_use,
                    temperature=temp_to_use,
                    driving_type=driving_type,
                    energy_used=energy_to_use,
                    elevation=elevation,
                    vehicle_load=load
                )

        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

        # Add labels
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_zlabel('Predicted Range (km)', fontsize=14)

        # Add title
        ax.set_title(
            f'3D Visualization of Range Predictions\n{factor1.capitalize()} vs {factor2.capitalize()} ({driving_type.capitalize()} driving)',
            fontsize=16, y=1.02)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Range (km)')

        plt.tight_layout()
        plt.savefig(f'3d_visualization_{factor1}_vs_{factor2}.png', dpi=300)
        plt.show()

        return X, Y, Z

    def create_heatmap(self, factor1='speed', factor2='temperature', energy_used=10,
                       driving_type='mixed', load=0, elevation=0):
        """
        Creates a heatmap showing how two factors interact to affect range.
        """
        if factor1 == factor2:
            print("Error: The two factors must be different")
            return

        # Define ranges for each factor
        speed_vals = np.linspace(10, 130, 25)
        temp_vals = np.linspace(-20, 40, 25)
        energy_vals = np.linspace(0, 70, 25)

        # Determine which values to use for each axis
        if factor1 == 'speed':
            x_vals = speed_vals
            x_label = 'Speed (km/h)'
        elif factor1 == 'temperature':
            x_vals = temp_vals
            x_label = 'Temperature (°C)'
        else:  # energy
            x_vals = energy_vals
            x_label = 'Energy Used (kWh)'

        if factor2 == 'speed':
            y_vals = speed_vals
            y_label = 'Speed (km/h)'
        elif factor2 == 'temperature':
            y_vals = temp_vals
            y_label = 'Temperature (°C)'
        else:  # energy
            y_vals = energy_vals
            y_label = 'Energy Used (kWh)'

        # Create data for heatmap
        data = np.zeros((len(y_vals), len(x_vals)))

        # Calculate range predictions for each combination
        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                # Set the parameters based on which factors we're using
                if factor1 == 'speed':
                    speed = x_val
                elif factor1 == 'temperature':
                    temperature = x_val
                else:  # energy
                    energy_used_val = x_val

                if factor2 == 'speed':
                    speed = y_val
                elif factor2 == 'temperature':
                    temperature = y_val
                else:  # energy
                    energy_used_val = y_val

                # Use default values for any parameter not specified by factor1 or factor2
                speed_to_use = speed if 'speed' in [factor1, factor2] else 60
                temp_to_use = temperature if 'temperature' in [factor1, factor2] else 20
                energy_to_use = energy_used_val if 'energy' in [factor1, factor2] else energy_used

                # Get prediction
                data[i, j] = self.predict_range(
                    speed=speed_to_use,
                    temperature=temp_to_use,
                    driving_type=driving_type,
                    energy_used=energy_to_use,
                    elevation=elevation,
                    vehicle_load=load
                )

        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(data, cmap='viridis', xticklabels=np.round(x_vals, 1),
                    yticklabels=np.round(y_vals, 1), annot=False, cbar_kws={'label': 'Range (km)'})

        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.title(
            f'Impact of {factor1.capitalize()} and {factor2.capitalize()} on Range\n({driving_type.capitalize()} driving, {energy_used} kWh used)',
            fontsize=16)

        # Improve x and y tick labels
        plt.xticks(np.linspace(0, len(x_vals) - 1, 10).astype(int),
                   np.round(x_vals[np.linspace(0, len(x_vals) - 1, 10).astype(int)], 1))
        plt.yticks(np.linspace(0, len(y_vals) - 1, 10).astype(int),
                   np.round(y_vals[np.linspace(0, len(y_vals) - 1, 10).astype(int)], 1))

        plt.tight_layout()
        plt.savefig(f'heatmap_{factor1}_vs_{factor2}.png', dpi=300)
        plt.show()

        return data

    def visualize_driving_type_comparison(self, temperature=20, energy_used=10,
                                          speed_range=(10, 130), load=0, elevation=0):
        """
        Compares range predictions for different driving types across speeds.
        """
        speeds = np.linspace(speed_range[0], speed_range[1], 25)

        plt.figure(figsize=(12, 8))

        for driving_type in ['city', 'highway', 'mixed']:
            ranges = []
            for speed in speeds:
                range_prediction = self.predict_range(
                    speed=speed,
                    temperature=temperature,
                    driving_type=driving_type,
                    energy_used=energy_used,
                    elevation=elevation,
                    vehicle_load=load
                )
                ranges.append(range_prediction)

            plt.plot(speeds, ranges, 'o-', linewidth=2, label=f'{driving_type.capitalize()} Driving')

        plt.title(
            f'Impact of Driving Type on Predicted Range\n(Temperature: {temperature}°C, Energy Used: {energy_used} kWh)',
            fontsize=16)
        plt.xlabel('Speed (km/h)', fontsize=14)
        plt.ylabel('Predicted Range (km)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('driving_type_comparison.png', dpi=300)
        plt.show()

        return speeds, ranges

    def visualize_load_impact(self, speed=60, temperature=20, driving_type='mixed',
                              energy_used=10, load_range=(0, 500), elevation=0):
        """
        Visualizes how vehicle load affects predicted range.
        """
        loads = np.linspace(load_range[0], load_range[1], 25)

        ranges = []
        for load in loads:
            range_prediction = self.predict_range(
                speed=speed,
                temperature=temperature,
                driving_type=driving_type,
                energy_used=energy_used,
                elevation=elevation,
                vehicle_load=load
            )
            ranges.append(range_prediction)

        plt.figure(figsize=(12, 8))
        plt.plot(loads, ranges, 'o-', linewidth=2, color='purple')
        plt.title(
            f'Impact of Vehicle Load on Predicted Range\n({driving_type.capitalize()} driving, {speed} km/h, {energy_used} kWh used)',
            fontsize=16)
        plt.xlabel('Vehicle Load (kg)', fontsize=14)
        plt.ylabel('Predicted Range (km)', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add annotations for common load scenarios
        plt.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
        plt.text(10, min(ranges), 'Driver only', fontsize=10)

        plt.axvline(x=150, linestyle='--', color='gray', alpha=0.5)
        plt.text(160, min(ranges), 'Driver + passenger', fontsize=10)

        plt.axvline(x=300, linestyle='--', color='gray', alpha=0.5)
        plt.text(310, min(ranges), 'Family (4 people)', fontsize=10)

        plt.axvline(x=450, linestyle='--', color='gray', alpha=0.5)
        plt.text(460, min(ranges), 'Family + luggage', fontsize=10)

        plt.tight_layout()
        plt.savefig('load_impact_visualization.png', dpi=300)
        plt.show()

        return loads, ranges

    def visualize_elevation_impact(self, speed=60, temperature=20, driving_type='mixed',
                                   energy_used=10, elevation_range=(-10, 10), load=0):
        """
        Visualizes how road gradient (elevation) affects predicted range.
        """
        elevations = np.linspace(elevation_range[0], elevation_range[1], 25)

        ranges = []
        for elevation in elevations:
            range_prediction = self.predict_range(
                speed=speed,
                temperature=temperature,
                driving_type=driving_type,
                energy_used=energy_used,
                elevation=elevation,
                vehicle_load=load
            )
            ranges.append(range_prediction)

        plt.figure(figsize=(12, 8))
        plt.plot(elevations, ranges, 'o-', linewidth=2, color='green')
        plt.title(
            f'Impact of Road Gradient on Predicted Range\n({driving_type.capitalize()} driving, {speed} km/h, {energy_used} kWh used)',
            fontsize=16)
        plt.xlabel('Road Gradient (%)', fontsize=14)
        plt.ylabel('Predicted Range (km)', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add annotations for reference
        plt.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
        plt.text(0.5, min(ranges), 'Flat road', fontsize=10)

        plt.axvline(x=5, linestyle='--', color='gray', alpha=0.5)
        plt.text(5.5, min(ranges), 'Moderate uphill', fontsize=10)

        plt.axvline(x=-5, linestyle='--', color='gray', alpha=0.5)
        plt.text(-4.5, min(ranges), 'Moderate downhill', fontsize=10)

        plt.tight_layout()
        plt.savefig('elevation_impact_visualization.png', dpi=300)
        plt.show()

        return elevations, ranges

    def run_comprehensive_test(self):
        """
        Runs a comprehensive test suite that demonstrates the model's performance
        across various conditions with multiple visualizations.
        """
        print("Running comprehensive test suite...\n")

        # Test 1: Speed Impact at Different Temperatures
        print("Test 1: Visualizing speed impact at different temperatures...")
        self.visualize_speed_impact(temperatures=[-10, 0, 20, 35], energy_used=10)

        # Test 2: Temperature Impact at Different Speeds
        print("\nTest 2: Visualizing temperature impact at different speeds...")
        self.visualize_temperature_impact(speeds=[30, 60, 100], energy_used=10)

        # Test 3: Battery Consumption Impact
        print("\nTest 3: Visualizing battery consumption impact...")
        self.visualize_battery_impact(speeds=[60], temperatures=[20])

        # Test 4: 3D Visualization of Speed vs Temperature
        print("\nTest 4: Creating 3D visualization of speed vs temperature...")
        self.create_3d_visualization(factor1='speed', factor2='temperature', energy_used=10)

        # Test 5: Heatmap of Speed vs Temperature
        print("\nTest 5: Creating heatmap of speed vs temperature...")
        self.create_heatmap(factor1='speed', factor2='temperature', energy_used=10)

        # Test 6: Driving Type Comparison
        print("\nTest 6: Comparing different driving types...")
        self.visualize_driving_type_comparison(temperature=20, energy_used=10)

        # Test 7: Vehicle Load Impact
        print("\nTest 7: Visualizing vehicle load impact...")
        self.visualize_load_impact(speed=60, temperature=20, energy_used=10)

        # Test 8: Elevation Impact
        print("\nTest 8: Visualizing elevation impact...")
        self.visualize_elevation_impact(speed=60, temperature=20, energy_used=10)

        print("\nComprehensive test suite completed!")


# Main function to run the tests
def main():
    print("=" * 80)
    print("EV RANGE PREDICTION MODEL - TESTING AND VISUALIZATION")
    print("=" * 80)

    # Initialize tester
    tester = EVModelTester()

    # Run comprehensive test suite
    tester.run_comprehensive_test()

    # You can also run individual tests as needed:
    # tester.visualize_speed_impact(temperatures=[-10, 0, 20, 35])
    # tester.create_heatmap(factor1='speed', factor2='energy')
    # tester.create_3d_visualization(factor1='temperature', factor2='energy')


if __name__ == "__main__":
    main()