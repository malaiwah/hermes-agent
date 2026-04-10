FROM debian:13.4


# Disable Python stdout buffering to ensure logs are printed immediately
ENV PYTHONUNBUFFERED=1

# ── Build metadata ───────────────────────────────────────────────────────────

ARG GIT_COMMIT=unknown
ARG GIT_REF=unknown
ARG BUILD_DATE=unknown
ARG IMAGE_SOURCE=http://10.15.0.6:3300/angelos/hermes-agent

LABEL org.opencontainers.image.title="hermes-agent" \
      org.opencontainers.image.description="Self-improving AI agent — creates skills from experience" \
      org.opencontainers.image.source="${IMAGE_SOURCE}" \
      org.opencontainers.image.base.name="debian:13.4" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.ref.name="${GIT_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

RUN printf 'IMAGE_TITLE=hermes-agent\nIMAGE_SOURCE=%s\nGIT_COMMIT=%s\nGIT_REF=%s\nBUILD_DATE=%s\n' \
    "${IMAGE_SOURCE}" "${GIT_COMMIT}" "${GIT_REF}" "${BUILD_DATE}" \
    > /etc/hermes-release

# ── System dependencies ──────────────────────────────────────────────────────

# Use apt-cacher-ng on the LAN for faster package downloads.
# The proxy is only used during build (not baked into the image).
ARG APT_PROXY=http://10.15.0.6:3142
RUN echo "Acquire::HTTP::Proxy \"${APT_PROXY}\";" > /etc/apt/apt.conf.d/01proxy && \
    apt-get update && \
    apt-get upgrade -y --target-release=stable-security && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 python3-pip ripgrep ffmpeg gcc \
        python3-dev libffi-dev podman-remote curl && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/apt/apt.conf.d/01proxy

# ── Install uv (much faster than pip for dependency resolution) ──────────────

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv

# ── Application source (includes oikos patches) ─────────────────────────────

COPY . /opt/hermes
WORKDIR /opt/hermes

RUN uv pip install --system --no-cache -e ".[all]" && \
    npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    cd /opt/hermes/scripts/whatsapp-bridge && \
    npm install --prefer-offline --no-audit && \
    npm cache clean --force

RUN chmod +x /opt/hermes/docker/entrypoint.sh \
             /opt/hermes/docker/wait-for-honcho.sh

# ── Podman-remote shim ───────────────────────────────────────────────────────
# Bridges DOCKER_HOST env var to podman-remote --url flag

RUN printf '#!/bin/sh\nprintf "%%s podman-remote-shim: %%s\n" "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)" "$*" >> /opt/data/logs/shim.log 2>/dev/null\nexec /usr/bin/podman-remote --url "${DOCKER_HOST:-unix:///var/run/docker.sock}" "$@"\n' \
    > /usr/local/bin/docker && chmod +x /usr/local/bin/docker

# ── Runtime ──────────────────────────────────────────────────────────────────

ENV HERMES_HOME=/opt/data
VOLUME ["/opt/data"]
ENTRYPOINT ["/opt/hermes/docker/wait-for-honcho.sh"]
