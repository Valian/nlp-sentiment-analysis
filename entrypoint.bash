#!/usr/bin/env bash

# we do this here to utilize volumes

if [ ! -f '/usr/local/lib/python3.5/dist-packages/en_vectors_web_lg/__init__.py' ]; then
    echo "Downloading spacy model, this may take a while..."
    python -m spacy download en_vectors_web_lg
fi


mkdir -p dist/data dist/models

exec $@