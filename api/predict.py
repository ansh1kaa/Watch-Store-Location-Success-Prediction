"""
Vercel Serverless API Handler for Watch Store Location Success Prediction
Pure Python HTTP Handler (No Flask / Django / Node.js needed).
Guarantees structured JSON output for both successful predictions and error handling.
Validates all inputs server-side with zero external stack-trace or path exposure.
Supports single-city name lookups mapped to synthetic academic location profiles.
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


# Synthetic Academic Location Profiles
SYNTHETIC_CITY_PROFILES = {
    'indore': {
        'name': 'Indore',
        'tagline': 'Commercial & Trade Hub (MG Road / Chappan Catchment)',
        'features': {
            'population': 280000,
            'average_monthly_income': 7200.0,
            'daily_foot_traffic': 15500,
            'nearby_competitors': 5,
            'monthly_rent': 9200.0,
            'distance_to_mall_km': 1.1,
            'nearby_retail_stores': 85,
            'estimated_monthly_customers': 2000,
            'average_purchase_value': 310.0,
            'monthly_operating_cost': 12500.0,
            'local_demand_score': 8.2,
            'target_age_group_score': 7.6
        }
    },
    'gwalior': {
        'name': 'Gwalior',
        'tagline': 'Outskirts Corridor (Low Traffic / High Rent Ratio)',
        'features': {
            'population': 95000,
            'average_monthly_income': 3400.0,
            'daily_foot_traffic': 3200,
            'nearby_competitors': 10,
            'monthly_rent': 9800.0,
            'distance_to_mall_km': 11.5,
            'nearby_retail_stores': 18,
            'estimated_monthly_customers': 450,
            'average_purchase_value': 140.0,
            'monthly_operating_cost': 11200.0,
            'local_demand_score': 3.5,
            'target_age_group_score': 3.8
        }
    },
    'delhi': {
        'name': 'Delhi',
        'tagline': 'Connaught Place / Prime Central Trade Area',
        'features': {
            'population': 420000,
            'average_monthly_income': 10200.0,
            'daily_foot_traffic': 23500,
            'nearby_competitors': 8,
            'monthly_rent': 15500.0,
            'distance_to_mall_km': 0.4,
            'nearby_retail_stores': 135,
            'estimated_monthly_customers': 3100,
            'average_purchase_value': 480.0,
            'monthly_operating_cost': 18000.0,
            'local_demand_score': 9.2,
            'target_age_group_score': 9.0
        }
    },
    'mumbai': {
        'name': 'Mumbai',
        'tagline': 'Bandra High-Street / Luxury Promenade',
        'features': {
            'population': 460000,
            'average_monthly_income': 11100.0,
            'daily_foot_traffic': 24000,
            'nearby_competitors': 9,
            'monthly_rent': 18500.0,
            'distance_to_mall_km': 0.3,
            'nearby_retail_stores': 140,
            'estimated_monthly_customers': 3300,
            'average_purchase_value': 520.0,
            'monthly_operating_cost': 21000.0,
            'local_demand_score': 9.5,
            'target_age_group_score': 9.1
        }
    },
    'bhopal': {
        'name': 'Bhopal',
        'tagline': 'MP Nagar Commercial Market Zone',
        'features': {
            'population': 210000,
            'average_monthly_income': 6200.0,
            'daily_foot_traffic': 12800,
            'nearby_competitors': 5,
            'monthly_rent': 6200.0,
            'distance_to_mall_km': 2.1,
            'nearby_retail_stores': 68,
            'estimated_monthly_customers': 1650,
            'average_purchase_value': 280.0,
            'monthly_operating_cost': 8600.0,
            'local_demand_score': 7.2,
            'target_age_group_score': 7.0
        }
    },
    'jaipur': {
        'name': 'Jaipur',
        'tagline': 'MI Road / Heritage Retail Belt',
        'features': {
            'population': 290000,
            'average_monthly_income': 7400.0,
            'daily_foot_traffic': 17200,
            'nearby_competitors': 6,
            'monthly_rent': 8200.0,
            'distance_to_mall_km': 1.5,
            'nearby_retail_stores': 95,
            'estimated_monthly_customers': 2250,
            'average_purchase_value': 360.0,
            'monthly_operating_cost': 10500.0,
            'local_demand_score': 8.1,
            'target_age_group_score': 8.0
        }
    },
    'pune': {
        'name': 'Pune',
        'tagline': 'FC Road / Tech Corridor Promenade',
        'features': {
            'population': 340000,
            'average_monthly_income': 8900.0,
            'daily_foot_traffic': 19500,
            'nearby_competitors': 7,
            'monthly_rent': 11000.0,
            'distance_to_mall_km': 0.8,
            'nearby_retail_stores': 110,
            'estimated_monthly_customers': 2600,
            'average_purchase_value': 410.0,
            'monthly_operating_cost': 13500.0,
            'local_demand_score': 8.7,
            'target_age_group_score': 8.5
        }
    }
}


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
        supported = [p['name'] for p in SYNTHETIC_CITY_PROFILES.values()]
        self._send_json(200, {
            "success": True,
            "status": "online",
            "message": "Watch Store Location Success Prediction API is operational.",
            "endpoint": "/api/predict",
            "supported_locations": supported,
            "disclosure": "Synthetic academic location profiles created for ML evaluation."
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

            # 1. Location name resolution (Indore, Gwalior, Delhi, etc.)
            if action == 'predict_location' or ('location' in payload and 'location_data' not in payload):
                loc_raw = payload.get('location', '')
                if not loc_raw or not isinstance(loc_raw, str):
                    self._send_json(400, {
                        "success": False,
                        "error": "Please provide a valid location name (e.g. 'Indore')."
                    })
                    return

                key = loc_raw.strip().lower()
                if key not in SYNTHETIC_CITY_PROFILES:
                    supported = [p['name'] for p in SYNTHETIC_CITY_PROFILES.values()]
                    self._send_json(400, {
                        "success": False,
                        "error": f"Location '{loc_raw.strip()}' is not in the synthetic academic profiles. Supported locations: {', '.join(supported)}.",
                        "supported_locations": supported
                    })
                    return

                profile_info = SYNTHETIC_CITY_PROFILES[key]
                canonical_name = profile_info['name']
                location_data = profile_info['features']

                pred_result = predict_module.predict_store_success(location_data, model_type=model_type)
                fin_result = predict_module.financial_investment_calculator(location_data)
                summary_text = predict_module.generate_business_summary(canonical_name, location_data, model_type=model_type)

                response_data = {
                    "success": True,
                    "action": "predict_location",
                    "location": canonical_name,
                    "location_name": canonical_name,
                    "tagline": profile_info.get('tagline', ''),
                    "location_profile": location_data,
                    "is_synthetic_profile": True,
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
                return

            elif action == 'compare':
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
