"""
Watch Store Location Success Prediction - Inference & Business Decision Module
Provides probability estimation, location comparison, sensitivity analysis, and financial heuristics.
"""

import os
import pickle
import pandas as pd
import numpy as np

def _get_model_file_path():
    """
    Finds trained_models.pkl reliably across local and Vercel serverless environments.
    """
    possible_paths = [
        # Strategy 1: Relative to src/
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "trained_models.pkl"),
        # Strategy 2: Relative to current working directory
        os.path.join(os.getcwd(), "models", "trained_models.pkl"),
        # Strategy 3: Relative to api/ directory if executed from api/
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "trained_models.pkl"),
        # Strategy 4: Vercel task root
        "/var/task/models/trained_models.pkl"
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
            
    raise FileNotFoundError(
        f"Trained model artifact (trained_models.pkl) not found. Checked: {possible_paths}"
    )

def _load_artifacts():
    model_path = _get_model_file_path()
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

def predict_store_success(location_data, model_type='rf'):
    """
    Predicts store profitability label and dynamic probability score (0-100%) using predict_proba().
    
    location_data: dict containing all 12 input features.
    model_type: 'rf' for Random Forest Classifier (default) or 'lr' for Logistic Regression.
    """
    artifacts = _load_artifacts()
    feature_names = artifacts['feature_names']
    
    # Ensure correct feature order
    input_df = pd.DataFrame([location_data])[feature_names]
    
    if model_type == 'lr':
        model = artifacts['lr_model']
        scaled_input = artifacts['scaler'].transform(input_df)
        pred_label = model.predict(scaled_input)[0]
        prob_positive = model.predict_proba(scaled_input)[0][1]
    else:
        model = artifacts['rf_model']
        pred_label = model.predict(input_df)[0]
        prob_positive = model.predict_proba(input_df)[0][1]
        
    status = "PROFITABLE" if pred_label == 1 else "NOT PROFITABLE"
    prob_percentage = round(float(prob_positive) * 100, 2)
    
    return {
        'prediction': status,
        'label': int(pred_label),
        'probability_percent': prob_percentage,
        'probability_raw': float(prob_positive),
        'model_used': 'Random Forest Classifier' if model_type == 'rf' else 'Logistic Regression'
    }

def compare_locations(locations_dict, model_type='rf'):
    """
    Ranks multiple proposed store candidate locations by model-estimated profitability probability.
    locations_dict: dict of {location_name: location_data_dict}
    """
    comparison_results = []
    for name, loc_data in locations_dict.items():
        res = predict_store_success(loc_data, model_type=model_type)
        comparison_results.append({
            'Location': name,
            'Prediction': res['prediction'],
            'Probability (%)': res['probability_percent'],
            'Rent ($)': loc_data.get('monthly_rent'),
            'Foot Traffic': loc_data.get('daily_foot_traffic'),
            'Competitors': loc_data.get('nearby_competitors')
        })
    
    comp_df = pd.DataFrame(comparison_results).sort_values(by='Probability (%)', ascending=False)
    return comp_df

def sensitivity_analysis(base_location_data, feature_name, value_list, model_type='rf'):
    """
    What-if sensitivity analysis: dynamically modifies a single feature and reruns model inference.
    """
    results = []
    for val in value_list:
        test_data = base_location_data.copy()
        test_data[feature_name] = val
        res = predict_store_success(test_data, model_type=model_type)
        results.append({
            feature_name: val,
            'Prediction': res['prediction'],
            'Probability (%)': res['probability_percent']
        })
    return pd.DataFrame(results)

def financial_investment_calculator(location_data, initial_setup_cost=150000, revenue_margin_factor=0.10):
    """
    Calculates post-prediction financial heuristics (Revenue, Costs, Profit, Break-even).
    
    NOTE: This financial calculator is a separate post-prediction business decision-support tool.
    It uses a 10% (0.10) effective retail margin assumption consistent with dataset baseline assumptions.
    Calculated values are NEVER fed into the ML model as input features.
    """
    est_cust = location_data['estimated_monthly_customers']
    avg_val = location_data['average_purchase_value']
    rent = location_data['monthly_rent']
    op_cost = location_data['monthly_operating_cost']
    
    # Financial estimate heuristics
    est_monthly_revenue = est_cust * avg_val * revenue_margin_factor
    total_monthly_cost = rent + op_cost
    est_monthly_net_profit = est_monthly_revenue - total_monthly_cost
    
    if est_monthly_net_profit > 0:
        breakeven_months = round(initial_setup_cost / est_monthly_net_profit, 1)
    else:
        breakeven_months = None  # Not profitable — break-even not applicable
        
    return {
        'est_monthly_revenue': round(est_monthly_revenue, 2),
        'total_monthly_cost': round(total_monthly_cost, 2),
        'est_monthly_net_profit': round(est_monthly_net_profit, 2),
        'initial_setup_cost': initial_setup_cost,
        'breakeven_months': breakeven_months
    }

def generate_business_summary(location_name, location_data, model_type='rf'):
    """
    Generates an executive summary report for a candidate store location.
    Qualitative badges (High, Moderate, Low, Strong, Average, Weak) are threshold-based business heuristics.
    """
    pred_res = predict_store_success(location_data, model_type=model_type)
    fin_res = financial_investment_calculator(location_data)
    
    # Simple threshold-based qualitative heuristics (not ML explanations)
    traffic_qual = "High" if location_data['daily_foot_traffic'] > 12000 else "Moderate" if location_data['daily_foot_traffic'] > 5000 else "Low"
    comp_qual = "High" if location_data['nearby_competitors'] > 8 else "Moderate" if location_data['nearby_competitors'] > 3 else "Low"
    rent_qual = "High" if location_data['monthly_rent'] > 10000 else "Moderate" if location_data['monthly_rent'] > 4000 else "Low"
    demand_qual = "Strong" if location_data['local_demand_score'] >= 7.0 else "Average" if location_data['local_demand_score'] >= 4.0 else "Weak"
    
    summary_text = f"""
==================================================
LOCATION ASSESSMENT: {location_name.upper()}
==================================================
Model Prediction: {pred_res['prediction']}
Probability of Profitability: {pred_res['probability_percent']}%
Model Selected: {pred_res['model_used']}

QUALITATIVE SITE METRICS (Business Threshold Heuristics):
- Daily Foot Traffic: {location_data['daily_foot_traffic']} ({traffic_qual})
- Nearby Competitors: {location_data['nearby_competitors']} ({comp_qual})
- Monthly Rent: ${location_data['monthly_rent']:,.2f} ({rent_qual})
- Local Demand Score: {location_data['local_demand_score']}/10 ({demand_qual})

SEPARATE FINANCIAL HEURISTICS (Academic Estimate):
- Projected Monthly Revenue: ${fin_res['est_monthly_revenue']:,.2f}
- Total Monthly Operating Cost: ${fin_res['total_monthly_cost']:,.2f}
- Estimated Monthly Net Profit: ${fin_res['est_monthly_net_profit']:,.2f}
- Estimated Break-Even Period: {fin_res['breakeven_months']} months

Disclaimer: Both the ML model probability and financial calculator are academic decision-support estimates and do not guarantee real-world store profitability.
==================================================
"""
    return summary_text.strip()

if __name__ == "__main__":
    sample_loc = {
        'population': 180000,
        'average_monthly_income': 6500.0,
        'daily_foot_traffic': 14000,
        'nearby_competitors': 3,
        'monthly_rent': 6500.0,
        'distance_to_mall_km': 1.2,
        'nearby_retail_stores': 45,
        'estimated_monthly_customers': 1200,
        'average_purchase_value': 320.0,
        'monthly_operating_cost': 7500.0,
        'local_demand_score': 8.2,
        'target_age_group_score': 7.5
    }
    
    print("Testing Single Location Prediction...")
    res = predict_store_success(sample_loc)
    print(f"Prediction: {res['prediction']}")
    print(f"Probability: {res['probability_percent']}%\n")
    
    print(generate_business_summary("Downtown Mall Site A", sample_loc))
