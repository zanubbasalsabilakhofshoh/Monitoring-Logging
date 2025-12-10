from prometheus_client import start_http_server, Gauge
import time
import random
import mlflow.pyfunc

# === METRIKS ===
latency_gauge = Gauge("model_latency", "Latency per prediction (ms)")
throughput_gauge = Gauge("model_throughput", "Predictions per second")
accuracy_gauge = Gauge("model_accuracy", "Model accuracy score")

# === LOAD MODEL ARTIFACT ===
model = mlflow.pyfunc.load_model("model")

def simulate_metrics():
    # simulasi latency Ms
    latency = random.uniform(30, 120)
    latency_gauge.set(latency)

    # throughput (pred/s)
    throughput = random.uniform(5, 20)
    throughput_gauge.set(throughput)

    # accuracy dummy (ubah sesuai artefak MLflow)
    accuracy = random.uniform(0.7, 0.95)
    accuracy_gauge.set(accuracy)

if __name__ == "__main__":
    print("Prometheus exporter berjalan di port: 9000")
    start_http_server(9000)

    while True:
        simulate_metrics()
        time.sleep(2)

