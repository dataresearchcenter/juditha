FROM ghcr.io/dataresearchcenter/ftmq:latest

COPY juditha /app/juditha
COPY setup.py /app/setup.py
COPY pyproject.toml /app/pyproject.toml
COPY VERSION /app/VERSION
COPY README.md /app/README.md

WORKDIR /app
# plyvel (via ftmq[level]) ships no wheel
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get update \
    && apt-get install -y --no-install-recommends libleveldb1d \
    && apt-get install -y --no-install-recommends g++ libleveldb-dev \
    && pip install "." \
    && apt-get purge -y --auto-remove g++ libleveldb-dev \
    && rm -rf /var/lib/apt/lists/*

USER 1000

# The gRPC api has no authentication.
ENV JUDITHA_RPC_HOST=0.0.0.0
# Mount a store built with `juditha build` here.
ENV JUDITHA_URI=/data/juditha.db

EXPOSE 50051

ENTRYPOINT ["juditha", "serve"]
