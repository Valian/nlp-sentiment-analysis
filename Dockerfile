FROM ubuntu:16.04

ENV PYTHON_PIP_VERSION 9.0.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev wget ca-certificates git libzmq-dev \
    && ln -s python3 /usr/bin/python \
    && wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py' \
    && python get-pip.py --disable-pip-version-check --no-cache-dir "pip==$PYTHON_PIP_VERSION" \
    && rm -rf /var/lib/apt/lists/* \
    && rm get-pip.py

WORKDIR /srv

RUN pip install --no-cache-dir  jupyter && \
    mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py

COPY requirements.txt /srv/
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# we use CPU version in this image
RUN pip install tensorflow

COPY entrypoint.bash /entrypoint.bash
ENTRYPOINT ["/entrypoint.bash"]

CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0"]