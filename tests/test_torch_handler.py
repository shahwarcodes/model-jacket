import torch
from runtime.handlers.torch_handler import TorchModelHandler
import tempfile

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x * 2

def test_torch_handler_prediction():
    model = DummyModel()
    scripted = torch.jit.script(model)
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        scripted.save(tmp.name)
        handler = TorchModelHandler(tmp.name)
        output = handler.predict([[1, 2, 3]])
        assert output == [[2, 4, 6]]
