import joblib
import numpy as np
from .base import BaseModelHandler

class SklearnModelHandler(BaseModelHandler):
    def load_model(self):
        self.model = joblib.load(self.model_path)

    def predict(self, input_data):
        array = np.asarray(input_data)
        output = self.model.predict(array)
        return output.tolist()
