from fastapi import FastAPI, HTTPException
from prometheus_client import Summary, Counter, Histogram, generate_latest, REGISTRY
import os, time
from runtime.api.schemas import InferenceRequest, InferenceResponse
from runtime.handlers.torch_handler import TorchModelHandler
from runtime.handlers.onnx_handler import ONNXModelHandler
from runtime.handlers.sklearn_handler import SklearnModelHandler

# Metrics
REQUEST_TIME = Histogram("inference_latency_seconds", "Inference latency in seconds")
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")

app = FastAPI()

# choose handler based on env
FRAMEWORK = os.getenv("MODEL_FRAMEWORK", "torch")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pt")

if FRAMEWORK == "onnx":
    handler = ONNXModelHandler(MODEL_PATH)
elif FRAMEWORK == "sklearn":
    handler = SklearnModelHandler(MODEL_PATH)
else:
    handler = TorchModelHandler(MODEL_PATH)

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return generate_latest(REGISTRY)

@app.post("/predict", response_model=InferenceResponse)
@REQUEST_TIME.time()
def predict(req: InferenceRequest):
    REQUEST_COUNT.inc()
    try:
        start = time.time()
        res = handler.predict(req.input_data)
        latency = (time.time() - start) * 1000
        return {"result": res, "latency_ms": latency}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
