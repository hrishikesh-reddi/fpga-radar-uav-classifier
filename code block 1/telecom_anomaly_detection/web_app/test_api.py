import requests
import json

# Test the main page
response = requests.get('http://127.0.0.1:5000/')
print("Main page status:", response.status_code)

# Test the generate data endpoint
payload = {
    "samples": 100
}

response = requests.post('http://127.0.0.1:5000/api/generate_data', 
                        json=payload,
                        headers={'Content-Type': 'application/json'})
print("Generate data endpoint status:", response.status_code)
if response.status_code == 200:
    data = response.json()
    print("Data summary:", data['summary'])

# Test the models endpoint
response = requests.get('http://127.0.0.1:5000/api/models')
print("Models endpoint status:", response.status_code)
if response.status_code == 200:
    models = response.json()
    print("Available models:", list(models.keys()))

# Test the detect anomalies endpoint
payload = {
    "model": "hybrid_qnn",
    "sample_size": 100
}

response = requests.post('http://127.0.0.1:5000/api/detect_anomalies', 
                        json=payload,
                        headers={'Content-Type': 'application/json'})
print("Detect anomalies endpoint status:", response.status_code)
if response.status_code == 200:
    result = response.json()
    print("Detection result:", result)