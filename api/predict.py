"""
Vercel Serverless API Endpoint for Watch Store Location Success Prediction
Pure Python HTTP Handler using standard library (No Flask/Django needed).
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Ensure src directory is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import (
    predict_store_success,
    financial_investment_calculator,
    generate_business_summary,
    compare_locations,
    sensitivity_analysis
)

class handler(BaseHTTPRequestHandler):

    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        self._set_headers(200)
        response = {
            "status": "online",
            "message": "Watch Store Location Success Prediction API is running.",
            "endpoint": "/api/predict",
            "disclosure": "Synthetic dataset created for academic machine-learning practice."
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            if not post_data:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Empty request body"}).encode('utf-8'))
                return

            payload = json.loads(post_data.decode('utf-8'))
            
            # Action router: 'predict' (default), 'compare', or 'sensitivity'
            action = payload.get('action', 'predict')

            if action == 'compare':
                locations = payload.get('locations', {})
                comp_df = compare_locations(locations)
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "action": "compare",
                    "comparison": comp_df.to_dict(orient='records')
                }).encode('utf-8'))
                return

            elif action == 'sensitivity':
                base_location = payload.get('location_data', {})
                feature_name = payload.get('feature_name', 'monthly_rent')
                value_list = payload.get('value_list', [4000, 6500, 9000, 12000, 15000])
                sens_df = sensitivity_analysis(base_location, feature_name, value_list)
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "action": "sensitivity",
                    "feature": feature_name,
                    "results": sens_df.to_dict(orient='records')
                }).encode('utf-8'))
                return

            else:
                # Single location prediction
                location_data = payload.get('location_data', payload)
                
                # Input validation: check required features
                required_features = [
                    'population', 'average_monthly_income', 'daily_foot_traffic',
                    'nearby_competitors', 'monthly_rent', 'distance_to_mall_km',
                    'nearby_retail_stores', 'estimated_monthly_customers',
                    'average_purchase_value', 'monthly_operating_cost',
                    'local_demand_score', 'target_age_group_score'
                ]
                
                missing = [f for f in required_features if f not in location_data]
                if missing:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({
                        "error": f"Missing required feature(s): {', '.join(missing)}"
                    }).encode('utf-8'))
                    return
                
                # Perform ML inference & heuristics
                model_type = payload.get('model_type', 'rf')
                pred_result = predict_store_success(location_data, model_type=model_type)
                fin_result = financial_investment_calculator(location_data)
                summary_text = generate_business_summary("Candidate Location", location_data, model_type=model_type)
                
                response_data = {
                    "prediction": pred_result['prediction'],
                    "label": pred_result['label'],
                    "probability_percent": pred_result['probability_percent'],
                    "probability_raw": pred_result['probability_raw'],
                    "model_used": pred_result['model_used'],
                    "financial_estimates": fin_result,
                    "summary_text": summary_text
                }

                self._set_headers(200)
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                "error": str(e),
                "message": "Internal server error during prediction."
            }).encode('utf-8'))
