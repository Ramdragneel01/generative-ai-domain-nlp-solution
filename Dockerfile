
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	UVICORN_HOST=0.0.0.0 \
	UVICORN_PORT=8003 \
	UVICORN_WORKERS=1 \
	LOG_LEVEL=info

WORKDIR /app

# Pull in latest security fixes available for the base image packages.
RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install --no-install-recommends -y ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser \
	&& chown -R appuser:appuser /app

USER appuser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
	CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=3).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn hallucination_lens.api:app --host ${UVICORN_HOST} --port ${UVICORN_PORT} --workers ${UVICORN_WORKERS} --log-level ${LOG_LEVEL}"]
