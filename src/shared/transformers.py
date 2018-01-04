import re

import numpy as np


class BaseTransformer(object):
    
    def fit(self, X, Y):
        return self

    def transform(self, X, Y=None):
        return X

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)
    
    
class RowTransformer(BaseTransformer):
    
    def transform(self, X, Y=None):
        return np.array(tuple(self.transform_value(X[i]) for i in range(X.shape[0])), dtype=X.dtype)
    
    def transform_value(self, value):
        return value
    

class ClearTextTransformer(RowTransformer):

    remove_html = re.compile(r"<[^>]*>")
    unwanted_characters = re.compile(r"[^\w+ !?']")
    merge_whitespaces = re.compile(r"\s\s+")
    
    def transform_value(self, value):
        value = self.remove_html.sub(' ', value)
        value = self.unwanted_characters.sub(' ', value)
        value = self.merge_whitespaces.sub(' ', value)
        return value.lower()


class NLPVectorTransformer(RowTransformer):

    def __init__(self, nlp):
        self.nlp = nlp

    def transform_value(self, value):
        return self.nlp(value).vector


class WordsToNlpIndexTransformer(RowTransformer):

    def __init__(self, nlp, remove_stop=False):
        self.remove_stop = remove_stop
        self.nlp = nlp

    def transform_value(self, value):
        processed_text = self.nlp(value)
        tokens = [w.lex_id for w in processed_text if ((not w.is_stop or not self.remove_stop) and w.is_alpha)]
        return np.array(tokens)


class NlpIndexToInputVectorTransformer(BaseTransformer):

    def __init__(self, nlp, padding_length):
        self.nlp = nlp
        self.padding_length = padding_length

    def transform(self, X, y=None):
        # here because it loads whole tensorflow
        from keras.preprocessing import sequence
        word_vectors = [list(self.sentence_to_vectors(sentence)) for sentence in X]
        word_vectors = sequence.pad_sequences(word_vectors, maxlen=self.padding_length, dtype='float32')
        return word_vectors

    def sentence_to_vectors(self, sentence):
        for word in sentence:
            try:
                yield self.nlp.vocab.vectors.data[word]
            except (IndexError, KeyError):
                pass
