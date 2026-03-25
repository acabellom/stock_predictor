FROM python:3.13-slim
 
WORKDIR /app
 
RUN pip install poetry
RUN pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cpu
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root
 
COPY . .

RUN pip install -e .

CMD ["uvicorn", "stock_predictor.predict:app", "--host", "0.0.0.0", "--port", "8000"]