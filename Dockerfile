
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

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

RUN pip install --no-cache-dir -e .

RUN useradd --create-home --uid 10001 appuser \
	&& chown -R appuser:appuser /app

USER appuser

EXPOSE 8003

CMD ["uvicorn", "hallucination_lens.api:app", "--host", "0.0.0.0", "--port", "8003"]
