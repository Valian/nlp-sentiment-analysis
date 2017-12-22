import hashlib
import math
import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files


def get_batch_slice(total_size, batch_size, batch_number):
    if batch_size < 0:
        batch_size = total_size
    number_of_batches = math.ceil(total_size / batch_size)
    batch_number %= number_of_batches
    start = batch_size * batch_number
    end = min(batch_size * (batch_number + 1), total_size)
    return slice(start, end)


class DataSet(object):

    name = None

    def __init__(self, data, metadata=None):
        """
        :type data: pd.DataFrame
        """
        self.data = data
        self.metadata = metadata or {}
        self.metadata['size'] = len(data)

    @property
    def size(self):
        return len(self.data)

    def get_data_and_labels(self, *, batch_slice):
        raise NotImplementedError()

    @classmethod
    def load_data(cls, path):
        raise NotImplementedError()

    def to_generator(self, batch_size):
        def generator():
            current_batch_number = 0
            while True:
                batch_slice = get_batch_slice(self.size, batch_size, current_batch_number)
                yield self.get_data_and_labels(batch_slice=batch_slice)
                current_batch_number += 1
        number_of_batches = math.ceil(self.size / batch_size)
        return number_of_batches, generator()

    def get_settings_hash(self):
        h = hashlib.md5()
        for key in sorted(self.metadata.keys()):
            val = self.metadata[key]
            h.update(str(key).encode())
            h.update(str(val).encode())
        return h

    def __getattr__(self, item):
        """Support for display"""
        if item.startswith("__"):
            raise AttributeError(item)
        return getattr(self.data, item)


class FineFoodReviewsDataSet(DataSet):

    name = 'food'

    def get_data_and_labels(self, *, batch_slice=None):
        text, labels = self.data['Text'], self.data['Score']
        if batch_slice:
            text, labels = text[batch_slice], labels[batch_slice]
        return text, labels

    def train_test_split(self, *, random_state=0, **kwargs):
        train, test = train_test_split(self.data, random_state=random_state, **kwargs)
        return self.__class__(train, self.metadata), self.__class__(test, self.metadata)

    @classmethod
    def load_data(cls, path, limit=-1, text_column='Summary'):
        connection = sqlite3.connect(path)
        query = """
            SELECT Score, lower({}) as Text
            FROM Reviews
            WHERE Score != 3
            """.format(text_column)
        if limit > 0:
            query += " LIMIT {}".format(limit)
        return cls(pd.read_sql_query(query, connection), metadata={
            'name': cls.name,
            'path': path,
            'limit': limit,
            'text_column': text_column
        })


class IMDBDataSet(DataSet):

    name = "IMDB"

    def get_data_and_labels(self, *, batch_slice=None):
        data = self.data
        if batch_slice:
            data = data[batch_slice]
        return data['text'], data['scores']

    @classmethod
    def load_data(cls, path, limit=-1):
        raw_data = load_files(path)
        data = pd.DataFrame({
            'text': pd.Series(raw_data['data']).astype(str),
            'scores': pd.Series(raw_data['target'])
        })
        data = data[data.scores < 2]
        if limit > 0:
            data = data[:limit]
        return cls(data, metadata={
            'name': cls.name,
            'path': path,
            'limit': limit
        })
