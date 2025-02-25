# ev-range-prediction# Enhanced Electric Vehicle Range Prediction Model

## Overview
The **Enhanced Electric Vehicle (EV) Range Prediction Model** is a machine learning system designed to accurately estimate the remaining driving range of an electric vehicle. It considers multiple factors such as speed, temperature, driving conditions, battery consumption, elevation, and vehicle load to produce reliable range predictions.

This model can be integrated into vehicle navigation systems, trip planning applications, or energy management solutions to help drivers optimize their EV usage.

---

## Features

### Primary Factors Modeled
- **Speed:** Models the impact of different speeds on energy consumption.
- **Temperature:** Considers ambient temperature effects on battery efficiency.
- **Driving Type:** Differentiates between city, highway, and mixed driving patterns.
- **Battery Consumption:** Tracks energy usage and its effect on range.
- **Vehicle Load:** Accounts for additional passenger and cargo weight.
- **Elevation:** Models uphill and downhill energy consumption effects.

### Advanced Features
- **Temperature-Energy Interaction:** Models temperature effects relative to battery depletion.
- **HVAC Impact Modeling:** Considers heating and air conditioning energy consumption.
- **Speed Efficiency Curve:** Uses a piecewise approach to model speed and energy consumption.
- **Combined Factor Handling:** Ensures realistic predictions when multiple conditions affect range.

---

## Model Architecture

### Core Components
1. **Gradient Boosting Regressor Model** – Learns complex patterns between input features and range outcomes.
2. **Feature Preprocessing System** – Transforms raw input data into engineered features.
3. **Feature Selection Module** – Identifies the most relevant variables for prediction.
4. **Data Standardization Component** – Ensures consistent handling of numerical inputs.

### Technical Specifications
- **Algorithm:** Gradient Boosting Regressor (scikit-learn implementation)
- **Features:** 14 selected from an initial pool of 28 engineered features
- **Training Data Size:** 15,000 synthetic data points
- **Hyperparameters:**
  - Learning rate: 0.05
  - Maximum tree depth: 5-7
  - Number of estimators: 300
  - Minimum samples split: 10
  - Subsample ratio: 0.8

### Performance Metrics
- **Mean Absolute Error (MAE):** ~4-5 kilometers
- **Root Mean Squared Error (RMSE):** ~5-6 kilometers
- **R² Score:** 0.996

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Required Packages:
  ```bash
  pip install numpy pandas scikit-learn joblib matplotlib seaborn
  ```

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ev-range-prediction.git
   cd ev-range-prediction
   ```
2. Place the model files in a directory named `model/`.
3. Verify installation by running:
   ```python
   import joblib
   model = joblib.load('model/enhanced_ev_model.joblib')
   print("Model loaded successfully!")
   ```

---

## Usage Guide

### Basic Range Prediction

```python
from model_utils import predict_range

optimal_range = predict_range(
    speed=60,
    temperature=20,
    driving_type='mixed',
    energy_used=5
)
print(f"Predicted Range: {optimal_range:.1f} km")
```

### Example Predictions
```python
# Winter highway driving
winter_range = predict_range(100, -10, 'highway', 15)
print(f"Winter highway range: {winter_range:.1f} km")
```

---

## Model Behavior

### Temperature Effects
- Maximum efficiency: ~20°C
- Cold weather (<0°C) reduces range by 20-30%
- Hot weather (>35°C) reduces range by 10-15%

### Speed Effects
- Peak efficiency: 30-60 km/h
- Range decreases significantly beyond 100 km/h

### Load & Elevation Effects
- Each 100 kg of extra weight reduces range by ~2%
- Uphill driving increases energy consumption
- Downhill driving allows some energy recovery

---

## Limitations & Future Improvements
### Current Limitations
- Based on synthetic data, not real-world measurements.
- Does not account for battery degradation over time.
- Extreme weather conditions may cause less accurate predictions.

### Future Enhancements
- Train with real-world EV data.
- Incorporate battery aging effects.
- Factor in traffic conditions and seasonal variations.

---

## References
1. **Synthetic Data Generation** - Based on EV physics principles.
2. **Gradient Boosting Regressor** - scikit-learn implementation.
3. **Feature Engineering** - Inspired by academic studies on EV energy consumption.

---

© 2025 Enhanced EV Range Prediction Model

