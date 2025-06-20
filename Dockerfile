FROM python:3.10-slim

WORKDIR /app

COPY runtime runtime
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

ENV MODEL_PATH=/models/model.pt
ENV MODEL_FRAMEWORK=torch

CMD ["uvicorn", "runtime.api.main:app", "--host", "0.0.0.0", "--port", "8000"]