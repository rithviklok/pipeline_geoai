# ── Dockerfile for Cloud Run ───────────────────────────────────────────────────
# Multi-stage build to keep the final image lean.
# No service-account key is baked in; ADC is injected by Cloud Run at runtime.

FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target /deps -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /deps /deps
COPY . .

ENV PYTHONPATH="/deps"
ENV PYTHONUNBUFFERED=1

# Cloud Run injects PORT; default to 8080 for local testing
ENV PORT=8080

EXPOSE ${PORT}

CMD exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}
