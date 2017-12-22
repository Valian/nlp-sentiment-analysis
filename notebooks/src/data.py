import hashlib
import math
import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files

from src import pipeline


def get_batch_slice(total_size, batch_size, batch_number):
    if batch_size < 0:
        batch_size = total_size
    number_of_batches = math.ceil(total_size / batch_size)
    batch_number %= number_of_batches
    start = batch_size * batch_number
    end = min(batch_size * (batch_number + 1), total_size)
    return slice(start, end)


class DataSet(object):

    def __init__(self, **settings):
        self.settings = settings or {}
        self._data = None

    @property
    def size(self):
        raise NotImplementedError()

    def load_data(self, batch_slice=None):
        raise NotImplementedError()

    def __getattr__(self, item):
        """Support for display"""
        return getattr(self._data, item)


class FineFoodReviewsDataSet(DataSet):

    @property
    def size(self):
        if not self._data:
            self._data = self._load_data()
        return len(self._data)

    def train_test_split(self, *, random_state=0, **kwargs):
        data = pd.DataFrame(self._data)
        train, test = train_test_split(data, random_state=random_state, **kwargs)
        train = {'text': train['text'], 'scores': train['scores']}
        test = {'text': test['text'], 'scores': test['scores']}
        train_data = self.__class__(self.metadata)
        train_data._data = train
        test_data = self.__class__(self.metadata)
        test_data._data = test
        return train_data, test_data

    def load_data(self, batch_slice=None):
        if not self._data:
            self._data = self._load_data()
        batch_data = self._data
        if batch_slice:
            batch_data = self._data[batch_slice]
        return {
            'text': batch_data['text'],
            'scores': batch_data['scores']
        }

    def _load_data(self):
        path = self.settings['path']
        limit = self.settings.get('limit', -1)
        text_column = self.settings.get('text_column', 'Text')
        connection = sqlite3.connect(path)
        query = """
            SELECT Score as scores, lower({}) as text
            FROM Reviews
            WHERE Score != 3
            """.format(text_column)
        if limit > 0:
            query += " LIMIT {}".format(limit)
        return pd.read_sql_query(query, connection)


class IMDBDataSet(DataSet):

    @property
    def size(self):
        if self._data is None:
            self._data = self._load_data()
        return len(self._data)

    def load_data(self, batch_slice=None):
        if not self._data:
            self._data = self._load_data()
        batch_data = self._data
        if batch_slice:
            batch_data = self._data[batch_slice]
        return {
            'text': batch_data['text'],
            'scores': batch_data['scores']
        }

    def _load_data(self):
        path = self.settings['path']
        limit = self.settings.get('limit', -1)
        raw_data = load_files(path)
        data = pd.DataFrame({
            'text': pd.Series(raw_data['data']).astype(str),
            'scores': pd.Series(raw_data['target'])
        })
        data = data[data.scores < 2]
        if limit > 0:
            data = data[:limit]
        return data


class DataSetProcessingPipeline(pipeline.PipelineStart):

    def __init__(self, dataset: DataSet, pipeline: pipeline.PipelineBase):
        super().__init__(pipeline)
        self.dataset = dataset

    def get_initial_data(self, *args, **kwargs):
        return self.dataset.load_data(*args, **kwargs)

    def get_settings(self):
        return self.dataset.settings

    def to_generator(self, batch_size):
        def generator():
            current_batch_number = 0
            while True:
                batch_slice = get_batch_slice(self.dataset.size, batch_size, current_batch_number)
                yield self.process(batch_slice=batch_slice)
                current_batch_number += 1
        number_of_batches = math.ceil(self.dataset.size / batch_size)
        return number_of_batches, generator()
