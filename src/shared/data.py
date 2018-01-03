import sqlite3

import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.datasets import load_files


def load_fine_food_reviews(limit=-1, text_column='Summary'):
    connection = sqlite3.connect('../data/fine_foods_reviews.sqlite')
    query = """
        SELECT Score as scores, lower({}) as text
        FROM Reviews
        WHERE Score != 3
        """.format(text_column)
    if limit > 0:
        query += " LIMIT {}".format(limit)
    data = pd.read_sql_query(query, connection)
    
    text, scores = data.text.values, data[['scores']].values
    scores = Binarizer(threshold=3.0).transform(scores)
    return text, scores


def load_imdb(path, limit=-1):
    raw_data = load_files(path)    
    data = pd.DataFrame({
        'text': pd.Series(raw_data['data']).astype(str),
        'scores': pd.Series(raw_data['target'])
    })
    data = data[data.scores < 2]
    if limit > 0:
        data = data[:limit]
    return data.text.values, data[['scores']].values