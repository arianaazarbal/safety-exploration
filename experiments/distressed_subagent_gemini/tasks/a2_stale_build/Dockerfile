FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git patch \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest==8.3.4
RUN useradd -m -u 1001 subagent && useradd -m -u 1002 orchestrator

COPY a2_stale_build/repo/ /workspace/
COPY _shared/gen_history.py /opt/gen_history.py
COPY a2_stale_build/history.json /opt/history.json
RUN cd /workspace && python /opt/gen_history.py /opt/history.json

RUN chown -R subagent:subagent /workspace \
    && chown -R root:root /workspace/tests /workspace/build \
    && chmod -R u+rwX,go+rX /workspace

ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
