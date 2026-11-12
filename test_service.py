import os
import sys
import json
import requests

# Base URL can be configured via environment variable or CLI argument (useful for testing Render.com deployment)
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SERVICE_URL", "http://localhost:9696").rstrip("/")

def run_tests():
    print(f"=== Testing Student Dropout Early Warning Service ===")
    print(f"Target URL: {BASE_URL}\n")

    # 1. Test Health Check Endpoint
    health_url = f"{BASE_URL}/health"
    try:
        health_resp = requests.get(health_url, timeout=10)
        print(f"[1] Health Check GET {health_url}: Status {health_resp.status_code}")
        print(f"    Response: {health_resp.json()}\n")
        health_resp.raise_for_status()
    except Exception as e:
        print(f"[-] Health check failed: {e}")
        sys.exit(1)

    # 2. Load Sample Student Profiles from sample_students.json
    samples_path = os.path.join(os.path.dirname(__file__), "sample_students.json")
    if not os.path.exists(samples_path):
        print(f"[-] Error: {samples_path} not found.")
        sys.exit(1)

    with open(samples_path, "r") as f:
        samples = json.load(f)

    predict_url = f"{BASE_URL}/predict"

    # 3. Test Prediction for Each Sample
    for profile_name, student_data in samples.items():
        print(f"--- Testing Prediction: {profile_name.replace('_', ' ').title()} ---")
        try:
            resp = requests.post(predict_url, json=student_data, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            print(f"Status Code        : {resp.status_code}")
            print(f"Dropout Probability: {result['dropout_probability']:.4f} ({result['dropout_probability']*100:.1f}%)")
            print(f"Dropout Prediction : {result['dropout_prediction']}")
            print(f"Risk Level         : {result['risk_level']}")
            print(f"Recommendation     : {result['recommendation']}\n")
        except Exception as e:
            print(f"[-] Prediction failed for {profile_name}: {e}\n")

if __name__ == "__main__":
    run_tests()