import os
import re
import copy

import numpy as np
import pandas as pd
from keras.preprocessing import sequence


class Middleware(object):

    def __call__(self, wrapped):
        new_wrapped = copy.copy(wrapped)
        for method_name in dir(self):
            wrapper = getattr(self, method_name)
            if method_name.startswith('wrap_'):
                wrapped_method_name = method_name[5:]
                original_method = getattr(wrapped, wrapped_method_name)
                wrapped_method = wrapper(original_method, original_object=wrapped)
                setattr(new_wrapped, wrapped_method_name, wrapped_method)
        return new_wrapped


class MiddlewareWithSettingsMixin(object):

    def wrap_get_settings_hash(self, handler, original_object):
        def wrapped(*args, **kwargs):
            h = handler(*args, **kwargs)
            settings = self.get_settings()
            for key in sorted(settings.keys()):
                val = settings[key]
                h.update(str(key).encode())
                h.update(str(val).encode())
            return h
        return wrapped

    def get_settings(self):
        return {}


class ModifyDataMiddleware(Middleware):

    def wrap_get_data_and_labels(self, handler, original_object):
        def wrapper(**kwargs):
            data, labels = handler(**kwargs)
            data = self.modify_data(data)
            labels = self.modify_labels(labels)
            return data, labels
        return wrapper

    def modify_data(self, data):
        return data

    def modify_labels(self, labels):
        return labels


class BinarizeLabelsMiddleware(MiddlewareWithSettingsMixin, ModifyDataMiddleware):

    def __init__(self, threshold):
        self.threshold = threshold

    def modify_labels(self, labels):
        def binarize_function(score):
            return int(score > self.threshold)
        return labels.apply(binarize_function)

    def get_settings(self):
        return {'threshold': self.threshold}


class ClearTextMiddleware(ModifyDataMiddleware):

    def modify_data(self, data):
        return data.apply(self.clear_function)

    @staticmethod
    def clear_function(text):
        only_alpha_regex = re.compile(r'[^\w+ ]')
        return only_alpha_regex.sub('', text)


class WordsToNlpIndexMiddleware(ModifyDataMiddleware):

    def __init__(self, nlp):
        self.nlp = nlp

    def modify_data(self, data):
        def text_to_tokens(text):
            processed_text = self.nlp(text)
            tokens = [w.lex_id for w in processed_text if (w.is_stop is False and str(w).isalnum())]
            return tokens
        return data.apply(text_to_tokens)


class CacheMiddleware(Middleware):

    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.data = None

    def wrap_get_data_and_labels(self, handler, original_object):
        settings_hash = original_object.get_settings_hash()
        cache_filename = self.get_filename(settings_hash)

        def wrapper(**kwargs):
            batch_slice = kwargs.pop('batch_slice', None)
            self.ensure_data_loaded(cache_filename, handler, **kwargs)
            if batch_slice:
                text, labels = self.data
                return text[batch_slice], labels[batch_slice]
            return self.data
        return wrapper

    def ensure_data_loaded(self, cache_filename, handler, **kwargs):
        if not self.data:
            try:
                self.data = self.load_data_from_file(cache_filename)
                print("Data loaded from {}".format(cache_filename))
            except:
                print("No cache found, generating data...")
                self.data = handler(**kwargs)
                self.save_data_to_file(self.data, cache_filename)
                print("Data generated and cached in {}".format(cache_filename))

    def get_filename(self, hash):
        return 'data_{}.h5'.format(hash.hexdigest()[:8])

    def load_data_from_file(self, filename):
        filepath = os.path.join(self.cache_path, filename)
        text = pd.read_hdf(filepath, 'text')
        labels = pd.read_hdf(filepath, 'labels')
        return text, labels

    def save_data_to_file(self, data, filename):
        filepath = os.path.join(self.cache_path, filename)
        text, labels = data
        text.to_hdf(filepath, 'text', mode='w')
        labels.to_hdf(filepath, 'labels', mode='a')


class NlpIndexToInputVectorMiddleware(MiddlewareWithSettingsMixin, ModifyDataMiddleware):

    def __init__(self, nlp, padding_length):
        self.nlp = nlp
        self.padding_length = padding_length

    def modify_data(self, data):
        word_vectors = [self.sentence_to_vectors(sentence) for sentence in data]
        padded_word_wectors = sequence.pad_sequences(word_vectors, maxlen=self.padding_length, dtype='float32')
        return padded_word_wectors

    def word_to_vector(self, word):
        try:
            return self.nlp.vocab.vectors.data[word]
        except IndexError:
            return None

    def sentence_to_vectors(self, sentence):
        word_vectors = [self.word_to_vector(word) for word in sentence]
        return [word_vector for word_vector in word_vectors if isinstance(word_vector, np.ndarray)]

    def get_settings(self):
        return {'padding_length': self.padding_length}
