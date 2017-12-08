FROM jupyter/base-notebook

RUN conda install -c conda-forge tensorflow keras pandas spacy
RUN python -m spacy  download en_vectors_web_lg
