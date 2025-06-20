import onnxruntime as ort
from .base import BaseModelHandler
import numpy as np

class ONNXModelHandler(BaseModelHandler):
    def load_model(self):
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_data):
        array = np.asarray(input_data).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: array})
        return outputs[0].tolist()