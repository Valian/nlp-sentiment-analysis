import os
import re
from _hashlib import HASH
from functools import wraps
from typing import List, Dict

import numpy as np
import pandas as pd

from src.common import get_hash_of_array, get_hash_of_dict


class PipelineBase(object):

    def create_pipeline(self, handler):
        return handler

    def get_settings_hash(self) -> HASH:
        raise NotImplementedError()


class PipelinePart(PipelineBase):

    def get_settings_hash(self) -> HASH:
        return get_hash_of_dict(self.get_settings())

    def get_settings(self):
        return {}


class Pipeline(PipelineBase):

    def __init__(self, pipeline_parts: List[PipelineBase]):
        self.pipeline_parts = pipeline_parts

    def create_pipeline(self, handler):
        for part in self.pipeline_parts:
            handler = part.create_pipeline(handler)
        return handler

    def get_settings_hash(self) -> HASH:
        hashes = [part.get_settings_hash() for part in self.pipeline_parts]
        return get_hash_of_array(hashes)


class ProcessPipelinePart(PipelinePart):

    def create_pipeline(self, handler):
        @wraps(handler)
        def wrapper(*args, **kwargs):
            data = handler(*args, **kwargs)
            return self.process(data)
        return wrapper

    def process(self, data):
        raise NotImplementedError()


class PipelineStart(PipelinePart):

    def __init__(self, pipeline: PipelineBase=None):
        self.pipeline = pipeline
        self._handler = self.create_pipeline()

    def get_initial_data(self, *args, **kwargs):
        raise NotImplementedError()

    def process(self, *args, **kwargs):
        return self._handler(*args, **kwargs)

    def create_pipeline(self, handler=None):
        if self.pipeline:
            return self.pipeline.create_pipeline(self.get_initial_data)
        else:
            return self.get_initial_data

    def get_settings_hash(self):
        h = super(PipelineStart, self).get_settings_hash()
        if self.pipeline:
            h.update(self.pipeline.get_settings_hash().digest())
        return h


class PipelineFunctionStart(PipelineStart):

    def __init__(self, method, pipeline):
        super().__init__(pipeline)
        self.method = method

    def get_initial_data(self, *args, **kwargs):
        return self.method(*args, **kwargs)


class PipelineValueStart(PipelineStart):

    def __init__(self, value, pipeline):
        super().__init__(pipeline)
        self.value = value

    def get_initial_data(self):
        return self.value


class SplitPipeline(PipelineBase):

    def __init__(self, pipelines_by_names: Dict[str, PipelineBase]):
        self.pipelines_by_names = pipelines_by_names

    @staticmethod
    def _get_part_of_input(name, input):
        def inner_wrapper(*args, **kwargs):
            return input.get(name)
        return inner_wrapper

    def create_pipeline(self, handler):
        @wraps(handler)
        def wrapper(*args, **kwargs):
            data = handler(*args, **kwargs)
            for name, part in self.pipelines_by_names.items():
                part_handler = part.create_pipeline(self._get_part_of_input(name, data))
                data[name] = part_handler(*args, **kwargs)
            return data
        return wrapper

    def get_settings_hash(self) -> HASH:
        return get_hash_of_dict({
            name: part.get_settings_hash() for name, part in self.pipelines_by_names.items()
        })


class BinarizeLabelsPipeline(ProcessPipelinePart):

    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, data):
        return data.apply(self.binarize_label)

    def binarize_label(self, score):
        return int(score > self.threshold)

    def get_settings(self):
        return {'threshold': self.threshold}


class ClearTextPipeline(ProcessPipelinePart):

    def process(self, data):
        return data.apply(self.clear_function)

    @staticmethod
    def clear_function(text):
        only_alpha_regex = re.compile(r'[^\w+ ]')
        return only_alpha_regex.sub('', text)


class WordsToNlpIndexPipeline(ProcessPipelinePart):

    def __init__(self, nlp):
        self.nlp = nlp

    def process(self, data):
        return data.apply(self.text_to_tokens)

    def text_to_tokens(self, text):
        processed_text = self.nlp(text)
        tokens = [w.lex_id for w in processed_text if (w.is_stop is False and str(w).isalnum())]
        return tokens


class CachePipeline(PipelinePart):

    def __init__(self, cache_path, pipeline: PipelineBase, invalidating_param=None):
        self.cache_path = cache_path
        self.pipeline = pipeline
        self.invalidating_param = invalidating_param
        self.data = {}

    def create_pipeline(self, handler):
        settings_hash = self.pipeline.get_settings_hash()

        def wrapper(*args, **kwargs):
            file_hash = self.calculate_file_hash(settings_hash, **kwargs)
            cache_filename = self.get_filename(file_hash)
            self.ensure_data_loaded(cache_filename, handler, *args, **kwargs)
            return self.data[cache_filename]
        return wrapper

    def calculate_file_hash(self, settings_hash, **kwargs):
        params = {}
        if self.invalidating_param:
            params = {self.invalidating_param: kwargs.get(self.invalidating_param, None)}
        args_hash = get_hash_of_dict(params)
        settings_hash.update(args_hash.digest())
        return settings_hash

    def ensure_data_loaded(self, cache_filename, handler, *args, **kwargs):
        if cache_filename not in self.data:
            try:
                self.data[cache_filename] = self.load_data_from_file(cache_filename)
                print("Data loaded from {}".format(cache_filename))
            except Exception:
                print("No cache found, generating data...")
                self.data[cache_filename] = handler(*args, **kwargs)
                self.save_data_to_file(self.data[cache_filename], cache_filename)
                print("Data generated and cached in {}".format(cache_filename))

    def get_filename(self, hash):
        return 'data_{}.h5'.format(hash.hexdigest()[:8])

    def load_data_from_file(self, filename):
        filepath = os.path.join(self.cache_path, filename)
        keys = pd.read_hdf(filepath, '_keys')
        return {key: pd.read_hdf(filepath, key) for key in keys}

    def save_data_to_file(self, data, filename):
        filepath = os.path.join(self.cache_path, filename)
        mode = 'w'
        for name, part in data.items():
            part.to_hdf(filepath, name, mode=mode)
            mode = 'a'
        keys = pd.Series(list(data.keys()))
        keys.to_hdf(filepath, '_keys', mode=mode)

    def get_settings_hash(self):
        h = super(CachePipeline, self).get_settings_hash()
        if self.pipeline:
            h.update(self.pipeline.get_settings_hash().digest())
        return h

    def get_settings(self):
        return {
            'cache_path': self.cache_path,
            'param': self.invalidating_param
        }


class NlpIndexToInputVectorPipeline(ProcessPipelinePart):

    def __init__(self, nlp, padding_length):
        self.nlp = nlp
        self.padding_length = padding_length

    def process(self, data):
        # here because it loads whole tensorflow
        from keras.preprocessing import sequence
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
