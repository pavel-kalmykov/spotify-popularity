FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv

WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --locked --no-dev

COPY predict.py ./
COPY --chown=10001:10001 models/model.joblib ./models/model.joblib
USER 10001
EXPOSE 9696
CMD [".venv/bin/python", "predict.py"]
