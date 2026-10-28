# Wind Tunnel web app: API + static frontend on one port.
#   docker build -t windtunnel .
#   docker run -p 8000:8000 -e NEBIUS_API_KEY=... -e TAVILY_API_KEY=... windtunnel
# Without NEBIUS_API_KEY it runs offline: real paper search, mock race, all labelled.
# Behind a reverse proxy (a PaaS load balancer, nginx, Cloudflare), add
#   -e WINDTUNNEL_TRUST_PROXY=1 [-e WINDTUNNEL_TRUSTED_PROXIES=<proxy CIDRs>] [-e WINDTUNNEL_PROXY_HOPS=<n>]
# so the rate limit sees client addresses. Never set it without a proxy in front: clients
# could then write X-Forwarded-For themselves and skip the rate limit.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WINDTUNNEL_TRUST_PROXY=0 \
    WINDTUNNEL_WARM_SAMPLES=1 \
    WINDTUNNEL_RUNS_PER_HOUR=5 \
    WINDTUNNEL_RUNS_PER_DAY=10 \
    WINDTUNNEL_RUN_BUDGET_USD=0.5 \
    WINDTUNNEL_MAX_IN_FLIGHT=10 \
    WINDTUNNEL_BUDGET_USD=10

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY windtunnel ./windtunnel
COPY web ./web
COPY tasks ./tasks
RUN uv sync --frozen --no-dev

# Reports, the HTTP cache and the spend ledger live in /app/.windtunnel.
# Mount a volume there to keep share links and the budget ledger across redeploys.
RUN useradd --create-home --uid 10001 app && mkdir -p /app/.windtunnel && chown -R app /app/.windtunnel
USER app
VOLUME ["/app/.windtunnel"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn windtunnel.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
