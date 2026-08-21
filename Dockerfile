FROM ghcr.io/dataresearchcenter/ftmq:latest

COPY juditha /app/juditha
COPY setup.py /app/setup.py
COPY pyproject.toml /app/pyproject.toml
COPY VERSION /app/VERSION
COPY README.md /app/README.md

WORKDIR /app
# Both workarounds below exist only because ftmq is pinned to a git branch
# instead of a release; drop them once it is back on PyPI.
#   - git: pip needs it to clone the branch
#   - pip uninstall: the base image already ships an ftmq of the same version
#     installed from a local path, which pip would otherwise treat as
#     satisfying the requirement and never fetch the branch
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip uninstall -y ftmq \
    && pip install "." \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

USER 1000

# The gRPC api has no authentication. Binding all interfaces is safe only
# because the container boundary decides what can reach it – do not publish
# this port to an untrusted network.
ENV JUDITHA_RPC_HOST=0.0.0.0
# Mount a store built with `juditha build` here.
ENV JUDITHA_URI=/data/juditha.db

EXPOSE 50051

ENTRYPOINT ["juditha", "serve"]
