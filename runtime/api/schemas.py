from pydantic import BaseModel
from typing import List, Any

class InferenceRequest(BaseModel):
    input_data: List[Any]

class InferenceResponse(BaseModel):
    result: Any
    latency_ms: float