from abc import ABC, abstractmethod

class BaseModelHandler(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model from disk into memory."""

    @abstractmethod
    def predict(self, input_data):
        """Perform inference and return results."""