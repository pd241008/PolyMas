FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir pipx && pipx ensurepath

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

COPY polymas_ml/ polymas_ml/
RUN pip install --no-cache-dir -e .

EXPOSE 50054
CMD ["python", "-m", "polymas_ml.serving"]
