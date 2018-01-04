import types
import copy
import math

import numpy as np
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier


class KerasBatchClassifier(KerasClassifier):
    """
    Use fit_generator in KerasClassifier to handle large data sets easily.
    """
    
    def __init__(self, build_fn=None, *, preprocess_pipeline=None, **sk_params):
        super().__init__(build_fn, **sk_params)
        self.preprocess_pipeline = preprocess_pipeline
        self.n_classes_ = 2
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, train_indices=None, test_indices=None, **kwargs):
        from keras.models import Sequential

        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit ################################################
        y = np.array(y)
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
        fit_args.update(kwargs)
        #############################################################################################################
        
        batch_size = self.sk_params["batch_size"]
        if test_indices is not None and train_indices is not None:
            train_generator = self.batch_generator(X[train_indices], y[train_indices], batch_size=batch_size)
            test_generator = self.batch_generator(X[test_indices], y[test_indices], batch_size=batch_size)
            self.__history = self.model.fit_generator(
                generator=train_generator,
                validation_data=test_generator,
                steps_per_epoch=math.ceil(len(train_indices) / batch_size),
                validation_steps=math.ceil(len(test_indices) / batch_size),
                **fit_args)
        else:
            train_generator = self.batch_generator(X, y, batch_size=batch_size)
            self.__history = self.model.fit_generator(
                generator=train_generator,
                steps_per_epoch=math.ceil(len(X) / batch_size),
                **fit_args)

        return self.__history

    def score(self, x, y, **kwargs):
        if self.preprocess_pipeline:
            x = self.preprocess_pipeline.transform(x)
        return super().score(x, y, **kwargs)
        
    def predict_proba(self, x, **kwargs):
        if self.preprocess_pipeline:
            x = self.preprocess_pipeline.transform(x)
        return super().predict_proba(x, **kwargs)
    
    def predict(self, x, **kwargs):
        if self.preprocess_pipeline:
            x = self.preprocess_pipeline.transform(x)
        return super().predict(x, **kwargs)        

    def batch_generator(self, x, y, batch_size=128):
        current_batch_number = 0        
        if self.preprocess_pipeline:
            self.preprocess_pipeline.fit(x)
        while True:
            batch_slice = self.get_batch_slice(x.size, batch_size, current_batch_number)
            batch_x = x[batch_slice]
            batch_y = y[batch_slice]
            if self.preprocess_pipeline:
                batch_x = self.preprocess_pipeline.transform(batch_x)
            yield batch_x, batch_y
            current_batch_number += 1
    
    @staticmethod
    def get_batch_slice(total_size, batch_size, batch_number):
        if batch_size < 0:
            batch_size = total_size
        number_of_batches = math.ceil(total_size / batch_size)
        batch_number %= number_of_batches
        start = batch_size * batch_number
        end = min(batch_size * (batch_number + 1), total_size)
        return slice(start, end)

    @property
    def history(self):
        return self.__history