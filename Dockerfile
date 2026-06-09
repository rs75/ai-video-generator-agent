# syntax=docker/dockerfile:1
#
# One image that runs BOTH halves of the pipeline:
#   - the Python ADK agent (Gemini / Imagen / Cloud TTS)
#   - Remotion (Node + headless Chrome) for the final MP4 render
#
# AUTH: nothing is baked in. At run time you mount the HOST's gcloud Application
# Default Credentials into the container, so it authenticates as "you" — exactly
# like it does on your machine. Run `gcloud auth application-default login` on
# the host once, then:
#
#   docker build -t video-agent .
#   docker run --rm -p 8080:8080 \
#     -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
#     video-agent
#
# Open http://localhost:8080 . Because we leave the standard ADC search path
# untouched, this SAME image also works unchanged on Cloud Run / GCE, where the
# credentials come from the metadata server instead of the mount.

FROM python:3.12-slim-bookworm

# --- System deps -------------------------------------------------------------
# Node 20 (runs the Remotion CLI) + the shared libraries headless Chrome needs
# to render, plus fonts so the caption text draws correctly.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg; \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
    apt-get install -y --no-install-recommends \
        nodejs \
        fonts-liberation fontconfig \
        libnss3 libdbus-1-3 libatk1.0-0 libgbm-dev libasound2 \
        libxrandr2 libxkbcommon-dev libxfixes3 libxcomposite1 libxdamage1 \
        libatk-bridge2.0-0 libpango-1.0-0 libcairo2 libcups2; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (the ADK agent) ---------------------------------------------
# Copied first so this layer is cached unless requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Remotion deps + bake its headless browser into the image ----------------
# Done before the full source copy so it caches on the lockfile, and so the
# first render doesn't pause to download Chrome at runtime.
COPY remotion/package.json remotion/package-lock.json ./remotion/
RUN cd remotion && npm ci && npx remotion browser ensure

# --- App source --------------------------------------------------------------
COPY . .
RUN mkdir -p workdir && chmod +x /app/docker-entrypoint.sh

# Vertex AI defaults. GOOGLE_CLOUD_PROJECT is intentionally NOT baked in — the
# entrypoint derives it from your mounted gcloud config at runtime (or pass
# `-e GOOGLE_CLOUD_PROJECT=...`), so no project id ships in the image.
ENV GOOGLE_GENAI_USE_VERTEXAI=TRUE \
    GOOGLE_CLOUD_LOCATION=us-central1

EXPOSE 8080
# The entrypoint resolves the GCP project from the mounted gcloud config, then
# runs the command below. adk web = the dev UI; bind 0.0.0.0 to reach it.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["adk", "web", "create_video_agent", "--host", "0.0.0.0", "--port", "8080"]
