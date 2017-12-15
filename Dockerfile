FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV PYTHON_PIP_VERSION 9.0.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev wget ca-certificates git libzmq-dev \
    && ln -s python3 /usr/bin/python \
    && wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py' \
    && python get-pip.py --disable-pip-version-check --no-cache-dir "pip==$PYTHON_PIP_VERSION" \
    && rm -rf /var/lib/apt/lists/* \
    && rm get-pip.py

WORKDIR /srv

COPY requirements.txt /srv/
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt
RUN python -m spacy download en_vectors_web_lg
RUN pip install jupyter && \
    mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py

COPY cudnn-8.0-linux-x64-v6.0.tgz /tmp/cudnn-8.0-linux-x64-v6.0.tgz
RUN tar zxvf /tmp/cudnn-8.0-linux-x64-v6.0.tgz -C /tmp && \
    mv /tmp/cuda/include/* /usr/local/cuda-8.0/include && \
    mv /tmp/cuda/lib64/* /usr/local/cuda-8.0/lib64
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
RUN pip install h5py tables

CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0"]