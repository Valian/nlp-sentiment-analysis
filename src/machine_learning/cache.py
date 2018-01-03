import h5py


class Cache(object):

    def __init__(self, path, X_dtype=None, y_dtype=None):
        self.path = path
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype

    def load(self):
        try:
            with h5py.File(self.path, 'r') as hf:
                return hf['X'][:], hf['y'][:]
        except KeyError as e:
            print("Unable to open cache, KeyError {}".format(e))
            return None
        except IOError:
            return None

    def save(self, X, y):
        with h5py.File(self.path, 'w') as hf:
            hf.create_dataset("X",  data=X, dtype=self.X_dtype or X.dtype)
            hf.create_dataset("y",  data=y, dtype=self.y_dtype or y.dtype)
