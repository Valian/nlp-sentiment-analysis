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

# Citations

<hr>

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}

<hr>

J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.