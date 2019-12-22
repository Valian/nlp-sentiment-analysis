import json
from functools import lru_cache

from shared import models


class Store(object):

    CONFIG_FILE = 'config.json'

    def __init__(self, nlp, config_filepath):
        self.config_filepath = config_filepath
        self.configuration = {}
        self.nlp = nlp
        self._cached_models = {}

    def load(self):
        with open(self.config_filepath, 'r') as f:
            self.configuration = json.load(f)

    def save(self):
        v = json.dumps(self.configuration, indent=2)
        with open(self.config_filepath, 'w') as f:
            f.write(v)

    def add_model(self, model, key, name, metadata=None):
        self.load()
        self.configuration[key] = {
            "name": name,
            "dataset_id": model.dataset_id,
            "class_name": model.__class__.__name__,
            "metadata": metadata or {},
            "model_params": model.model_params
        }
        self.save()

    def remove_model(self, name):
        self.load()
        del self.configuration[name]
        self.save()

    @lru_cache()
    def get_model(self, name):
        cfg = self.configuration[name]
        model_class = getattr(models, cfg['class_name'])
        model = model_class(self.nlp, cfg['dataset_id'], **cfg['model_params'])
        model.load()
        return model
