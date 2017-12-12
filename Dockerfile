FROM jupyter/base-notebook

RUN conda install -c conda-forge tensorflow keras pandas spacy
RUN python -m spacy  download en_vectors_web_lg
RUN conda install scikit-learn
RUN conda install matplotlib
RUN pip install sklearn-pandas