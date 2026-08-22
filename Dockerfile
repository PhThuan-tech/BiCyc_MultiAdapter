ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG APP_USER=researcher
ARG APP_UID=1000
ARG APP_GID=1000
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/workspace/.cache/huggingface TORCH_HOME=/workspace/.cache/torch
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends git tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid ${APP_GID} ${APP_USER} \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash ${APP_USER}
COPY requirements/docker.txt requirements/base.txt /tmp/requirements/
RUN python -m pip install --upgrade "pip==24.3.1" && python -m pip install -r /tmp/requirements/docker.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps . \
    && mkdir -p data outputs checkpoints .cache/huggingface .cache/torch \
    && chown -R ${APP_UID}:${APP_GID} /workspace
USER ${APP_USER}
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
