"""
Vercel Serverless API Handler for Watch Store Location Success Prediction
Pure Python HTTP Handler (No Flask / Django / Node.js needed).
Guarantees structured JSON output for both successful predictions and error handling.
Validates all inputs server-side with zero external stack-trace or path exposure.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import math

# Ensure src directory is in Python path for Vercel Serverless environment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Explicitly load src/predict.py avoiding namespace collision with api/predict.py
predict_module = None
import_error = None

def _load_predict_module():
    global import_error
    import importlib.util
    candidate_paths = [
        os.path.join(SRC_DIR, "predict.py"),
        os.path.join(PROJECT_ROOT, "src", "predict.py"),
        "/var/task/src/predict.py"
    ]
    for path in candidate_paths:
        abs_p = os.path.abspath(path)
        if os.path.exists(abs_p):
            try:
                spec = importlib.util.spec_from_file_location("watch_store_engine", abs_p)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["watch_store_engine"] = mod
                spec.loader.exec_module(mod)
                return mod
            except Exception as e:
                import_error = str(e)
                return None
    # Fallback attempt
    try:
        from src import predict as p
        return p
    except Exception as e:
        import_error = str(e)
        return None

predict_module = _load_predict_module()


REQUIRED_FEATURES = [
    'population', 'average_monthly_income', 'daily_foot_traffic',
    'nearby_competitors', 'monthly_rent', 'distance_to_mall_km',
    'nearby_retail_stores', 'estimated_monthly_customers',
    'average_purchase_value', 'monthly_operating_cost',
    'local_demand_score', 'target_age_group_score'
]

FEATURE_CONSTRAINTS = {
    'population': (1, 50_000_000, False),
    'average_monthly_income': (1, 1_000_000, False),
    'daily_foot_traffic': (0, 1_000_000, True),
    'nearby_competitors': (0, 5_000, True),
    'monthly_rent': (1, 2_000_000, False),
    'distance_to_mall_km': (0, 500, True),
    'nearby_retail_stores': (0, 10_000, True),
    'estimated_monthly_customers': (0, 500_000, True),
    'average_purchase_value': (0.01, 100_000, False),
    'monthly_operating_cost': (0, 2_000_000, True),
    'local_demand_score': (0.0, 10.0, True),
    'target_age_group_score': (0.0, 10.0, True)
}


def validate_location_data(data):
    """
    Validates location features server-side.
    Checks: dictionary format, missing keys, null/empty values, types, and ranges.
    Returns: (is_valid, cleaned_dict_or_error_message)
    """
    if not isinstance(data, dict):
        return False, "Location data must be an object with feature key-value pairs."

    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return False, f"Missing required feature(s): {', '.join(missing)}"

    cleaned = {}
    for f in REQUIRED_FEATURES:
        val = data[f]
        if val is None or val == "":
            return False, f"Feature '{f}' cannot be empty."

        try:
            num_val = float(val)
        except (ValueError, TypeError):
            return False, f"Feature '{f}' must be a valid number."

        if math.isnan(num_val) or math.isinf(num_val):
            return False, f"Feature '{f}' must be a finite numeric value."

        min_v, max_v, allow_zero = FEATURE_CONSTRAINTS[f]
        if allow_zero:
            if num_val < min_v or num_val > max_v:
                return False, f"Feature '{f}' must be between {min_v} and {max_v}."
        else:
            if num_val <= 0 or num_val > max_v:
                return False, f"Feature '{f}' must be greater than 0 and up to {max_v}."

        if f in ('population', 'daily_foot_traffic', 'nearby_competitors', 'nearby_retail_stores', 'estimated_monthly_customers'):
            cleaned[f] = int(round(num_val))
        else:
            cleaned[f] = float(num_val)

    return True, cleaned


def safe_json_value(val):
    """Convert numpy/non-standard types to JSON-safe Python primitives."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    try:
        import numpy as np
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            f = float(val)
            if math.isinf(f) or math.isnan(f):
                return None
            return f
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, np.bool_):
            return bool(val)
    except ImportError:
        pass
    return val


def sanitize_dict(d):
    """Recursively sanitize a dictionary or list for JSON serialization."""
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [sanitize_dict(item) for item in d]
    return safe_json_value(d)


class handler(BaseHTTPRequestHandler):

    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, status_code, data):
        """Safe JSON response writer. Guarantees valid JSON output for every response."""
        self._set_headers(status_code)
        sanitized = sanitize_dict(data)
        self.wfile.write(json.dumps(sanitized, default=str).encode('utf-8'))

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        self._send_json(200, {
            "success": True,
            "status": "online",
            "message": "Watch Store Location Success Prediction API is operational.",
            "endpoint": "/api/predict",
            "disclosure": "Synthetic dataset created for academic machine-learning practice."
        })

    def do_POST(self):
        try:
            if predict_module is None:
                self._send_json(500, {
                    "success": False,
                    "error": "Server error: Prediction engine could not be initialized."
                })
                return

            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self._send_json(400, {
                    "success": False,
                    "error": "Empty request payload received."
                })
                return

            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {
                    "success": False,
                    "error": "Malformed JSON in request body: Unable to parse JSON payload."
                })
                return

            if not isinstance(payload, dict):
                self._send_json(400, {
                    "success": False,
                    "error": "Request body must be a JSON object."
                })
                return

            action = payload.get('action', 'predict')
            raw_model_type = payload.get('model_type', 'rf')
            model_type = 'lr' if str(raw_model_type).lower() in ('lr', 'logistic', 'logistic_regression') else 'rf'

            if action == 'compare':
                locations = payload.get('locations')
                if not locations or not isinstance(locations, dict):
                    self._send_json(400, {
                        "success": False,
                        "error": "Candidate locations must be provided as a non-empty dictionary."
                    })
                    return

                cleaned_locations = {}
                for name, candidate in locations.items():
                    valid, result_or_err = validate_location_data(candidate)
                    if not valid:
                        self._send_json(400, {
                            "success": False,
                            "error": f"Invalid data for candidate '{name}': {result_or_err}"
                        })
                        return
                    cleaned_locations[name] = result_or_err

                comp_df = predict_module.compare_locations(cleaned_locations, model_type=model_type)
                self._send_json(200, {
                    "success": True,
                    "action": "compare",
                    "model_used": "Random Forest Classifier" if model_type == 'rf' else "Logistic Regression",
                    "comparison": comp_df.to_dict(orient='records')
                })
                return

            elif action == 'sensitivity':
                base_location = payload.get('location_data')
                valid, result_or_err = validate_location_data(base_location)
                if not valid:
                    self._send_json(400, {
                        "success": False,
                        "error": f"Base location invalid: {result_or_err}"
                    })
                    return

                feature_name = payload.get('feature_name', 'monthly_rent')
                if feature_name not in REQUIRED_FEATURES:
                    self._send_json(400, {
                        "success": False,
                        "error": f"Invalid feature '{feature_name}' for sensitivity analysis."
                    })
                    return

                value_list = payload.get('value_list')
                if not isinstance(value_list, list) or len(value_list) == 0:
                    self._send_json(400, {
                        "success": False,
                        "error": "value_list must be a non-empty array of numbers."
                    })
                    return

                cleaned_values = []
                for val in value_list:
                    try:
                        num_val = float(val)
                        if math.isnan(num_val) or math.isinf(num_val):
                            raise ValueError
                        cleaned_values.append(num_val)
                    except (ValueError, TypeError):
                        self._send_json(400, {
                            "success": False,
                            "error": f"Invalid value '{val}' in value_list: must be numeric."
                        })
                        return

                sens_df = predict_module.sensitivity_analysis(result_or_err, feature_name, cleaned_values, model_type=model_type)
                self._send_json(200, {
                    "success": True,
                    "action": "sensitivity",
                    "feature": feature_name,
                    "model_used": "Random Forest Classifier" if model_type == 'rf' else "Logistic Regression",
                    "results": sens_df.to_dict(orient='records')
                })
                return

            else:
                raw_loc = payload.get('location_data', payload)
                valid, result_or_err = validate_location_data(raw_loc)
                if not valid:
                    self._send_json(400, {
                        "success": False,
                        "error": result_or_err
                    })
                    return

                location_data = result_or_err
                pred_result = predict_module.predict_store_success(location_data, model_type=model_type)
                fin_result = predict_module.financial_investment_calculator(location_data)
                summary_text = predict_module.generate_business_summary("Candidate Location", location_data, model_type=model_type)

                response_data = {
                    "success": True,
                    "prediction": pred_result['prediction'],
                    "probability": round(pred_result['probability_raw'], 4),
                    "probability_percent": pred_result['probability_percent'],
                    "probability_raw": pred_result['probability_raw'],
                    "label": pred_result['label'],
                    "model_used": pred_result['model_used'],
                    "financial_estimates": fin_result,
                    "summary_text": summary_text
                }

                self._send_json(200, response_data)

        except Exception as err:
            # Internal logging without leaking filesystem or stack traces to HTTP client
            sys.stderr.write(f"API Internal Error: {err}\n")
            self._send_json(500, {
                "success": False,
                "error": "An internal server error occurred while processing the prediction request."
            })
