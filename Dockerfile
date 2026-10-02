# Two images from one file:
#   api  the FastAPI service with the four MCP servers in-process
#   llm  llama.cpp's OpenAI-compatible server, the same pinned CPU build the benchmark uses;
#        it downloads its model into a volume on first start

FROM python:3.12-slim AS api
WORKDIR /app
RUN useradd --create-home --uid 10001 app
COPY pyproject.toml LICENSE NOTICE ./
COPY src ./src
COPY tasks ./tasks
RUN pip install --no-cache-dir ".[agent,service]"
ENV SHOPFLOOR_DATA_DIR=/data \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data && chown app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=30 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
# Build the plant database on first start (pinned, checksummed download), then serve.
CMD ["sh", "-c", "test -s /data/plant.db || shopfloor data; exec shopfloor serve --host 0.0.0.0 --port 8000"]

FROM ubuntu:24.04 AS llm
ARG LLAMA_CPP=b11327
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libcurl4 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/llama \
 && curl -sSfL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP}/llama-${LLAMA_CPP}-bin-ubuntu-x64.tar.gz" \
    | tar -xz -C /opt/llama \
 && ln -s "$(dirname "$(find /opt/llama -name llama-server -type f | head -1)")" /opt/llama/bin
ENV LD_LIBRARY_PATH=/opt/llama/bin \
    MODEL_FILE=/models/model.gguf
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=60 CMD curl -sf http://127.0.0.1:8080/health
ENTRYPOINT ["sh", "-c", "test -s \"$MODEL_FILE\" || curl -sSfL --retry 3 -o \"$MODEL_FILE\" \"$MODEL_URL\"; exec /opt/llama/bin/llama-server -m \"$MODEL_FILE\" --jinja -c 16384 -np 1 --cache-reuse 256 --host 0.0.0.0 --port 8080"]
