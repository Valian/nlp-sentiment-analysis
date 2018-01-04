from functools import wraps

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_training(history):
    plt.plot(history.history['loss'], c='red')
    try:
        _ = plt.plot(history.history['val_loss'], c='blue')
    except:
        pass


def pandas_settings(**settings):
    settings = {k.replace('__', '.'): v for k, v in settings.items()}

    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            prev_settings = {key: pd.get_option(key) for key in settings}
            try:
                for k, v in settings.items():
                    pd.set_option(k, v)
                return f(*args, **kwargs)
            finally:
                for k, v in prev_settings.items():
                    pd.set_option(k, v)
        return inner
    return decorator
