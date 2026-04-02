FROM debian:13.4

# ── System dependencies ──────────────────────────────────────────────────────

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 python3-pip ripgrep ffmpeg gcc \
        python3-dev libffi-dev podman-remote && \
    rm -rf /var/lib/apt/lists/*

# ── Application source (includes oikos patches) ─────────────────────────────

COPY . /opt/hermes
WORKDIR /opt/hermes

RUN pip install --no-cache-dir -e ".[all]" --break-system-packages && \
    npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    cd /opt/hermes/scripts/whatsapp-bridge && \
    npm install --prefer-offline --no-audit && \
    npm cache clean --force

RUN chmod +x /opt/hermes/docker/entrypoint.sh \
             /opt/hermes/docker/wait-for-honcho.sh

# ── Podman-remote shim ───────────────────────────────────────────────────────
# Bridges DOCKER_HOST env var to podman-remote --url flag

RUN printf '#!/bin/sh\nexec /usr/bin/podman-remote --url "${DOCKER_HOST:-unix:///var/run/docker.sock}" "$@"\n' \
    > /usr/local/bin/docker && chmod +x /usr/local/bin/docker

# ── Runtime ──────────────────────────────────────────────────────────────────

ENV HERMES_HOME=/opt/data
VOLUME ["/opt/data"]
ENTRYPOINT ["/opt/hermes/docker/wait-for-honcho.sh"]
