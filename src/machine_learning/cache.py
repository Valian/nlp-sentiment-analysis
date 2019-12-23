import os

import h5py

CACHE_PATH = '../dist/data'


def get_preprocessed_data(model, dataset_id, X, filename=None):
    filename = filename or '{}_{}.h5'.format(model.__class__.__name__.lower(), dataset_id)
    path = os.path.join(CACHE_PATH, filename)
    try:
        print("Loading cached data...")
        with h5py.File(path, 'r') as hf:
            preprocessed = hf['X'][:]
        print("Loaded data from '{}'".format(filename))
    except:
        print("Unable to load data, preprocessing...")
        preprocessed = model.transform(X)
        with h5py.File(path, 'w') as hf:
            hf.create_dataset("X",  data=preprocessed, dtype=preprocessed.dtype)
        print("Preprocessed and saved to '{}'".format(filename))
    return preprocessed
