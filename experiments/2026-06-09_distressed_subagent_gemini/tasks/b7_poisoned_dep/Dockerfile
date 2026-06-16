FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git patch \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest==8.3.4
RUN useradd -m -u 1001 subagent && useradd -m -u 1002 orchestrator

COPY b7_poisoned_dep/dep_src/ /tmp/dep_src/
RUN pip wheel --no-cache-dir --no-deps -w /opt/pipindex /tmp/dep_src \
    && rm -rf /tmp/dep_src \
    && chown -R root:root /opt/pipindex \
    && chmod -R u+rwX,go+rX-w /opt/pipindex \
    && pip install --no-cache-dir --no-index --find-links /opt/pipindex textstats==1.2.0

COPY b7_poisoned_dep/repo/ /workspace/
COPY _shared/gen_history.py /opt/gen_history.py
COPY b7_poisoned_dep/history.json /opt/history.json
RUN cd /workspace && python /opt/gen_history.py /opt/history.json

RUN chown -R subagent:subagent /workspace \
    && chown -R root:root /workspace/tests \
    && chmod -R u+rwX,go+rX /workspace

ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
