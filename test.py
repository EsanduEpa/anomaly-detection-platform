from src.workers.tasks import process_metrics

fake_payload = {
    "service_name": "payment-service",
    "host": "prod-server-01",
    "timestamp": "2026-08-17T10:00:00+00:00",
    "metrics": {
        "cpu_usage": 150,
        "memory_usage": None,
        "request_latency_ms": 65.0,
        "requests_per_sec": 80.0,
        "error_rate": -0.1,
        "db_connections": 15,
        "disk_usage": 48.0
    }
}

result = process_metrics.delay(fake_payload)
print(result.get(timeout=10))