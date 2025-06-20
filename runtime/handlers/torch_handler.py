import torch
from .base import BaseModelHandler

class TorchModelHandler(BaseModelHandler):
    def load_model(self):
        self.model = torch.jit.load(self.model_path)
        self.model.eval()

    def predict(self, input_data):
        with torch.no_grad():
            tensor = torch.tensor(input_data)
            output = self.model(tensor)
            return output.tolist()