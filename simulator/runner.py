import time
import requests
import json
from simulator.generator import get_metrics_payload

# The URL of our FastAPI server
API_URL = "http://127.0.0.1:8080/api/v1/metrics"

# Simulated services and their servers
SERVICES = [
    {"service_name": "payment-service", "host": "prod-server-01"},
    {"service_name": "user-service",    "host": "prod-server-02"},
    {"service_name": "api-gateway",     "host": "prod-server-03"},
]

def send_metrics(service: dict):
    """Sends one batch of metrics for one service to the API."""
    payload = get_metrics_payload(
        service_name=service["service_name"],
        host=service["host"],
        inject_anomaly=True  # 10% chance of injecting an anomaly
    )

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code == 202:
            # Show the CPU value so we can see what was sent
            cpu = payload["metrics"]["cpu_usage"]
            print(f"✅ [{service['service_name']}] CPU: {cpu:.1f}% → Saved")
        else:
            print(f"❌ [{service['service_name']}] Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")


def run():
    """Main loop — sends metrics for all services every 5 seconds."""
    print("🚀 Simulator started. Sending metrics every 5 seconds...")
    print("   Press CTRL+C to stop.\n")

    while True:
        for service in SERVICES:
            send_metrics(service)
        print("---")
        time.sleep(5)  # Wait 5 seconds before next batch


if __name__ == "__main__":
    run()