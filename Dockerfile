FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y sudo wget ca-certificates git build-essential gnupg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/
RUN uv python install -i /tmp/python-download 3.12.11 \
    && cp -r /tmp/python-download/cpython-*/* /usr/local/ \
    && rm -rf /tmp/python-download \
    && rm /usr/local/lib/python3.12/EXTERNALLY-MANAGED

RUN uv pip install --system numpy tqdm scipy matplotlib

RUN usermod -l docker ubuntu
RUN usermod -d /home/docker -m docker
RUN echo 'docker ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/docker

# ffmpeg install for mp4 video generation
# RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg && \
#     rm -rf /var/lib/apt/lists/*

# openfoam install
# RUN wget -qO- https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc && \
#    echo "deb http://dl.openfoam.org/ubuntu noble main" | tee /etc/apt/sources.list.d/openfoam.list && \
#    apt-get update && \
#    apt-get install -y --no-install-recommends openfoam13 paraview && \
#    rm -rf /var/lib/apt/lists/*
# example command: `cd cavity && blockMesh && foamRun`

USER docker
WORKDIR /workspace

ENTRYPOINT []
CMD ["/bin/bash"]
