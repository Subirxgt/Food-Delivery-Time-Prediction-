import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_confidence_interval(model, final_input, confidence=0.95):
    """Calculate confidence interval for predictions"""
    # For ensemble models like Random Forest, we can use prediction variance
    if hasattr(model, 'estimators_'):
        # Get predictions from all trees
        predictions = np.array([tree.predict(final_input) for tree in model.estimators_])
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Calculate confidence interval
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z_score * std_pred
        
        lower_bound = mean_pred - margin
        upper_bound = mean_pred + margin
        
        return (max(0, float(lower_bound)), float(upper_bound))
    else:
        # For other models, use a simple heuristic
        prediction = model.predict(final_input)[0]
        margin = prediction * 0.15  # 15% margin
        return (max(0, prediction - margin), prediction + margin)

def generate_prediction_insights(input_data, prediction):
    """Generate insights about the prediction"""
    insights = []
    
    # Distance impact
    distance = input_data['distance_km'].iloc[0]
    if distance > 15:
        insights.append(f"Long distance ({distance:.1f} km) is a major factor in delivery time.")
    elif distance < 3:
        insights.append(f"Short distance ({distance:.1f} km) helps reduce delivery time.")
    
    # Weather impact
    weather = input_data['Weatherconditions'].iloc[0]
    if weather in ['Stormy', 'Sandstorms']:
        insights.append(f"Adverse weather ({weather}) may increase delivery time.")
    elif weather == 'Sunny':
        insights.append("Good weather conditions favor faster delivery.")
    
    # Traffic impact
    traffic = input_data['Road_traffic_density'].iloc[0]
    if traffic in ['High', 'Jam']:
        insights.append(f"Heavy traffic ({traffic}) is likely to delay delivery.")
    elif traffic == 'Low':
        insights.append("Low traffic conditions help maintain optimal delivery time.")
    
    # Time of day
    hour = input_data['order_hour'].iloc[0]
    if 12 <= hour <= 14 or 19 <= hour <= 21:
        insights.append(f"Peak ordering time ({hour}:00) may affect delivery speed.")
    
    # Multiple deliveries
    multi_del = input_data['multiple_deliveries'].iloc[0]
    if multi_del > 1:
        insights.append(f"Multiple deliveries ({multi_del}) will increase total time.")
    
    # Weekend effect
    is_weekend = input_data['is_weekend'].iloc[0]
    if is_weekend:
        insights.append("Weekend orders may have different delivery patterns.")
    
    # Vehicle type
    vehicle = input_data['Type_of_vehicle'].iloc[0]
    if vehicle in ['Bicycle', 'Electric Bike']:
        insights.append(f"{vehicle} may be slower but more eco-friendly.")
    elif vehicle == 'Motorcycle':
        insights.append("Motorcycle delivery offers good speed and flexibility.")
    
    # Festival impact
    festival = input_data['Festival'].iloc[0]
    if festival == 'Yes':
        insights.append("Festival season may affect delivery times due to increased demand.")
    
    return insights

def calculate_delivery_statistics(history_data):
    """Calculate statistics from prediction history"""
    if not history_data:
        return {}
    
    predictions = [record['prediction'] for record in history_data]
    
    stats = {
        'total_predictions': len(predictions),
        'average_time': np.mean(predictions),
        'median_time': np.median(predictions),
        'std_deviation': np.std(predictions),
        'min_time': np.min(predictions),
        'max_time': np.max(predictions)
    }
    
    return stats

def analyze_prediction_trends(history_data):
    """Analyze trends in prediction history"""
    if len(history_data) < 2:
        return {}
    
    # Sort by timestamp
    sorted_data = sorted(history_data, key=lambda x: x['timestamp'])
    
    # Calculate trends
    predictions = [record['prediction'] for record in sorted_data]
    timestamps = [record['timestamp'] for record in sorted_data]
    
    # Simple trend analysis
    if len(predictions) >= 5:
        recent_avg = np.mean(predictions[-5:])
        earlier_avg = np.mean(predictions[:-5])
        trend = "increasing" if recent_avg > earlier_avg else "decreasing"
    else:
        trend = "stable"
    
    # Time between predictions
    time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 60 
                  for i in range(1, len(timestamps))]
    avg_interval = np.mean(time_diffs) if time_diffs else 0
    
    return {
        'trend': trend,
        'average_interval_minutes': avg_interval,
        'prediction_frequency': len(predictions)
    }

def generate_recommendations(input_data, prediction):
    """Generate recommendations to optimize delivery time"""
    recommendations = []
    
    # Distance-based recommendations
    distance = input_data['distance_km'].iloc[0]
    if distance > 10:
        recommendations.append("Consider using a faster vehicle for long distances.")
    
    # Weather-based recommendations
    weather = input_data['Weatherconditions'].iloc[0]
    if weather in ['Stormy', 'Sandstorms', 'Fog']:
        recommendations.append("Allow extra time for adverse weather conditions.")
        recommendations.append("Consider rescheduling during severe weather.")
    
    # Traffic-based recommendations
    traffic = input_data['Road_traffic_density'].iloc[0]
    if traffic in ['High', 'Jam']:
        recommendations.append("Use traffic-aware routing to avoid congestion.")
        recommendations.append("Consider delivery during off-peak hours.")
    
    # Time-based recommendations
    hour = input_data['order_hour'].iloc[0]
    if 12 <= hour <= 14 or 19 <= hour <= 21:
        recommendations.append("Peak hours - consider pre-positioning delivery partners.")
    
    # Vehicle recommendations
    vehicle = input_data['Type_of_vehicle'].iloc[0]
    if vehicle == 'Bicycle' and distance > 5:
        recommendations.append("Consider upgrading to a motorized vehicle for efficiency.")
    
    # Multiple delivery optimization
    multi_del = input_data['multiple_deliveries'].iloc[0]
    if multi_del > 2:
        recommendations.append("Optimize delivery route to minimize total time.")
    
    return recommendations

def calculate_efficiency_score(input_data, prediction):
    """Calculate an efficiency score for the delivery"""
    base_time = input_data['distance_km'].iloc[0] * 2  # Baseline: 2 minutes per km
    
    # Adjust for various factors
    weather_multiplier = {
        'Sunny': 1.0, 'Cloudy': 1.1, 'Windy': 1.2, 
        'Stormy': 1.4, 'Sandstorms': 1.5, 'Fog': 1.3
    }
    
    traffic_multiplier = {
        'Low': 1.0, 'Medium': 1.2, 'High': 1.4, 'Jam': 1.8
    }
    
    vehicle_multiplier = {
        'Motorcycle': 1.0, 'Scooter': 1.1, 'Electric Bike': 1.2, 'Bicycle': 1.5
    }
    
    weather = input_data['Weatherconditions'].iloc[0]
    traffic = input_data['Road_traffic_density'].iloc[0]
    vehicle = input_data['Type_of_vehicle'].iloc[0]
    
    expected_time = base_time * weather_multiplier.get(weather, 1.2) * \
                   traffic_multiplier.get(traffic, 1.2) * \
                   vehicle_multiplier.get(vehicle, 1.2)
    
    # Add preparation time
    expected_time += input_data['prep_time_min'].iloc[0]
    
    # Calculate efficiency score (lower is better)
    efficiency_score = min(100, max(0, 100 - abs(prediction - expected_time) / expected_time * 100))
    
    return efficiency_score
