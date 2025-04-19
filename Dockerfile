FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG target=/mdt/run

WORKDIR ${target}
ARG DEBIAN_FRONTEND=noninteractive
ADD https://bootstrap.pypa.io/get-pip.py /tmp/get-pip.py
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 && \
  apt update && \
  apt install -y --no-install-recommends libasound2 python3.12 && \
  cat /tmp/get-pip.py | python3.12 && \
  pip3.12 install --no-cache-dir poetry==2.0.1 && \
  poetry config virtualenvs.create false

ADD sherpa/pyproject.toml sherpa/poetry.lock ${target}/

RUN poetry install --no-cache --no-root --with gpu

COPY sherpa/pb/ ${target}/pb/
COPY sherpa/*.py ${target}/

RUN python3.12 -m compileall ${target}
CMD ["python3.12", "server.py"]
EXPOSE 18913
