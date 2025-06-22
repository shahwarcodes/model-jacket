# Model Jacket

[![PyPI version](https://img.shields.io/pypi/v/model-jacket.svg)](https://pypi.org/project/model-jacket/)
[![Build](https://github.com/shahwarcodes/model-jacket/actions/workflows/publish.yml/badge.svg)](https://github.com/shahwarcodes/model-jacket/actions/workflows/publish.yml)
[![License](https://img.shields.io/github/license/shahwarcodes/model-jacket)](./LICENSE)

**Model Jacket** is a lightweight toolkit that wraps any ML model (Torch, ONNX, Scikit-learn) into a scalable, observable API server with one command. It provides Dockerized model APIs, metrics, Kubernetes deployment, and a PyPI-installable CLI — all in one place.

---

## 🚀 Features

- 🐳 **One-command containerization** with `jacket build`
- 🔌 **Framework-agnostic**: PyTorch, ONNX, Scikit-learn supported
- ⚡ **FastAPI-based async inference** with Uvicorn
- 📊 **Built-in Prometheus metrics** and `/healthz` endpoint
- ☸️ **Helm chart** for scalable K8s deployment + HPA
- ✅ **CI/CD to PyPI + GHCR** on every Git tag push
- 🧪 Includes **batching**, versioning, and logging hooks

---

## 🧑‍💻 Quick Start

### 1. Train and Export a Model
```python
# export_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier().fit(X, y)
joblib.dump(clf, "model.pkl")
```

### 2. Build the Container
```bash
jacket build --framework sklearn --model-path model.pkl --tag iris-v1
```

### 3. Run Locally
```bash
docker run -p 8000:8000 model-jacket:iris-v1
```

### 4. Call the API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": [[5.1, 3.5, 1.4, 0.2]]}'
```

---

## ☁️ Deploy with Helm
```bash
helm install iris ./helm/model-jacket \
  --set image.repository=ghcr.io/shahwarcodes/model-jacket \
  --set image.tag=iris-v1
```

---

## 📦 Install from PyPI
```bash
pip install model-jacket
```

---

## 📁 Project Structure
```
model-jacket/
├── jacket_cli/         # CLI for build & serve
├── runtime/            # FastAPI server + handlers
│   ├── api/
│   └── handlers/
├── helm/               # Helm chart for Kubernetes
├── .github/workflows/  # CI/CD pipeline
├── Dockerfile
├── setup.py
└── README.md
```

---

## 📈 Observability

- `/metrics`: Prometheus-compatible metrics (latency, QPS)
- `/healthz`: Liveness & readiness probes
- Structured logs with model version + timing info

---

## 🔁 Versioning & Shadow Testing

- Track every model as `model:v1`, `model:v1.1`, etc.
- Route production traffic to `v1`, and shadow to `v2`
- Compare predictions and latency without user exposure
- Rollback safely using Helm or container tags

---

## 🤝 Contributing

Contributions welcome! Feel free to fork, submit a PR, or open issues.

---

## 📄 License

[MIT](./LICENSE)