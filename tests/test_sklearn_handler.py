import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from runtime.handlers.sklearn_handler import SklearnModelHandler
import tempfile

def test_sklearn_handler_prediction():
    model = RandomForestClassifier(n_estimators=1)
    X = [[1, 2], [2, 3], [3, 4]]
    y = [0, 1, 0]
    model.fit(X, y)
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        joblib.dump(model, tmp.name)
        handler = SklearnModelHandler(tmp.name)
        result = handler.predict([[1, 2]])
        assert result in [[0], [1]]  # Tree might vary