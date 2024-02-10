FROM python:3.10.12-bullseye

ENV TZ=Asia/Shanghai

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    # 安装基础包
    apt update && \
    apt install -y \
    wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev graphviz \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    # 安装Poetry
    pip install poetry && \
    poetry config virtualenvs.in-project true && \
    rm -r /root/.cache/pip

WORKDIR /workspace