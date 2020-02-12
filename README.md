# Sentiment analysis experiments

This repository contains code for performing experiments checking quality of sentiment analysis of different ML algorithms and datasets.

# Used datasets
* [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
* [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)

# Setup

To test notebooks, you need these datasets unpacked and saved to 'data' directory:

[Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)
[IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)

In notebooks, I assume following files in data directory:

```bash
$ cd data
$ ls
fine_foods_reviews.sqlite  # renamed from database.sqlite
aclImdb                    # extracted IMDB dataset
```

<hr>

This project is designed to be run using docker. CPU version is much easier to run:

```bash
# we'll creating symlink to CPU docker-compose to be used by default
ln -s docker-compose.yml docker-compose.cpu.yml
docker-compose up

# jupyter notebook should be listening on localhost:8888, with all examples ready to be run.
```

GPU version requires nvidia-docker and nvidia-docker-compose. After installing, run:

```bash
# we'll creating symlink to GPU docker-compose to be used by default
ln -s docker-compose.yml docker-compose.gpu.yml

docker-compose build

# run just one time to ensure proper volumes are created
nvidia-docker run --rm nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04 echo "It's working"

# start using nvidia-docker-compose
nvidia-docker-compose up

```

After opening `localhost:8888`, there should be a file browser allowing to run `Jupyter Notebooks`.

# Repository structure

Notebooks used for testing: 

1. IMDB Reviews models generation and results: `src/text_sentiment_analysis_food.ipynb`
2. Amazon Fine Food Reviews models generation and results: `src/text_sentiment_analysis_imdb.ipynb`
3. Results comparision: `src/results_comparision.ipynb`

Web application code is in `src/app.py`. 

There is a quite large code base shared between notebooks and application, available inside `src/shared` directory.
Model definitions can be found in `src/shared/models.py`. 