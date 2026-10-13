FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.lock.txt pyproject.toml ./
COPY src ./src
# Optional public CA bundle for builds behind a trusted enterprise HTTPS proxy.
# The bundle is mounted for installation only; verification remains enabled.
RUN --mount=type=secret,id=build_ca,target=/run/secrets/build_ca \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    pip install --no-cache-dir --require-hashes -r requirements.lock.txt && \
    pip install --no-cache-dir --no-deps .
COPY app ./app
COPY configs ./configs
COPY sql ./sql
COPY scripts ./scripts
EXPOSE 8501
# Mount trusted locally generated artifacts at /app/artifacts before starting.
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.headless=true"]
