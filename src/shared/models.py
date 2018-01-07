import os

from keras.layers import Dense, Conv1D, BatchNormalization, GlobalMaxPooling1D, Dropout, MaxPooling1D
from keras.models import Sequential, load_model, save_model
from keras.regularizers import l2
from sklearn import pipeline
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from shared import transformers
from shared.common import get_hash_of_dict
from shared.batch_classifier import KerasBatchClassifier

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_DIRECTORY = os.path.realpath(os.path.join(CURRENT_DIR, '../../dist/models'))


class Model(object):

    def __init__(self, dataset_id, **model_params):
        self.dataset_id = dataset_id
        self.model_params = model_params
        self._pipeline = None

    def __getattr__(self, item):
        return getattr(self._pipeline, item)

    @property
    def file_ext(self):
        raise NotImplementedError()

    @property
    def model(self):
        return self._pipeline.steps[-1][1]

    @model.setter
    def model(self, value):
        self._pipeline.steps[-1] = ('model', value)

    @property
    def name(self):
        try:
            return getattr(self, 'NAME')
        except AttributeError:
            raise NotImplementedError("You need to define name of a model")

    @property
    def filename(self):
        h = get_hash_of_dict(self.model_params)
        return '{}_{}_{}.{}'.format(self.name, self.dataset_id, h.hexdigest()[:8], self.file_ext)

    def load(self, directory=DEFAULT_MODEL_DIRECTORY):
        raise NotImplementedError()

    def save(self, directory=DEFAULT_MODEL_DIRECTORY):
        raise NotImplementedError()

    def train(self, X, y, **train_params):
        raise NotImplementedError()

    def load_or_train(self, X, y, directory=DEFAULT_MODEL_DIRECTORY, **train_params):
        try:
            print("Loading model...")
            self.load(directory)
            print("Model '{}' loaded".format(self.filename))
        except IOError:
            print("Unable to load model, training...")
            self.train(X, y, **train_params)
            self.save(directory)
            print("Model '{}' saved".format(self.filename))
        return self

    def summary(self):
        raise NotImplementedError()


class SklearnModel(Model):

    def __init__(self, nlp, dataset_id, **model_params):
        super().__init__(dataset_id, **model_params)
        self.nlp = nlp
        self._pipeline = pipeline.Pipeline([
            ('clear', transformers.ClearTextTransformer()),
            ('nlp', transformers.NLPVectorTransformer(self.nlp)),
            ('model', None)
        ])

    @property
    def file_ext(self):
        return 'pkl'

    def train(self, X, y, preprocessed=False, **train_params):
        self.model = self.create_model(**self.model_params)
        if not preprocessed:
            self.fit(X, y, **train_params)
        else:
            self.model.fit(X, y, **train_params)
        return self._pipeline

    def save(self, directory=DEFAULT_MODEL_DIRECTORY):
        filepath = os.path.join(directory, self.filename)
        joblib.dump(self.model, filepath)

    def load(self, directory=DEFAULT_MODEL_DIRECTORY):
        filepath = os.path.join(directory, self.filename)
        self.model = joblib.load(filepath)
        return self

    def summary(self):
        return self.model

    def create_model(self, **model_params):
        raise NotImplementedError()

    @classmethod
    def from_sklearn_model(cls, model_class):
        name = model_class.__name__.replace('Classifier', '') + 'Model'
        return type(name, (cls,), {
            'NAME': model_class.__name__.lower(),
            'create_model': model_class,
        })


def _transform_identity(X, y=None):
    return X


DecisionTreeModel = SklearnModel.from_sklearn_model(DecisionTreeClassifier)
MLPModel = SklearnModel.from_sklearn_model(MLPClassifier)
GaussianNBModel = SklearnModel.from_sklearn_model(GaussianNB)
AdaBoostModel = SklearnModel.from_sklearn_model(AdaBoostClassifier)
RandomForestModel = SklearnModel.from_sklearn_model(RandomForestClassifier)
GradientBoostingModel = SklearnModel.from_sklearn_model(GradientBoostingClassifier)
QuadraticDiscriminantAnalysisModel = SklearnModel.from_sklearn_model(QuadraticDiscriminantAnalysis)
LogisticRegressionModel = SklearnModel.from_sklearn_model(LogisticRegression)
SVCModel = SklearnModel.from_sklearn_model(SVC)


class KerasModel(Model):

    NAME = 'keras'

    def __init__(self, nlp, dataset_id, **model_params):
        model_params.setdefault('epochs', 20)
        model_params.setdefault('batch_size', 128)
        model_params.setdefault('max_words_in_sentence', 200)
        super().__init__(dataset_id, **model_params)
        self.nlp = nlp
        self.max_words_in_sentence = model_params['max_words_in_sentence']
        self._pipeline = pipeline.Pipeline([
            ('clear', transformers.ClearTextTransformer()),
            ('nlp_index', transformers.WordsToNlpIndexTransformer(self.nlp)),
            ('model', None)
        ])

    @property
    def file_ext(self):
        return 'h5'

    def load(self, directory=DEFAULT_MODEL_DIRECTORY):
        filepath = os.path.join(directory, self.filename)
        model = self.create_model(**self.model_params)
        model.model = load_model(filepath=filepath)
        self.model = model
        return self

    def train(self, X, y, train_indices=None, test_indices=None, preprocessed=False):
        self.model = self.create_model(**self.model_params)
        if not preprocessed:
            self.fit(X, y, model__train_indices=train_indices, model__test_indices=test_indices)
        else:
            self.model.fit(X, y, train_indices=train_indices, test_indices=test_indices)
        return self._pipeline

    def save(self, directory=DEFAULT_MODEL_DIRECTORY):
        filepath = os.path.join(directory, self.filename)
        save_model(self.model.model, filepath)

    def create_model(self, **model_params):
        return KerasBatchClassifier(
            build_fn=self._build_conv1d,
            preprocess_pipeline=pipeline.Pipeline([
                ('nlp_input', transformers.NlpIndexToInputVectorTransformer(
                    self.nlp, self.max_words_in_sentence))
            ]),
            **model_params)

    def summary(self):
        return self.model.model.summary()

    @staticmethod
    def _build_conv1d(max_words_in_sentence=200, embedding_dim=300, filters=32, kernel_size=5, l2_weight=0.001,
                     dropout_rate=0.7):
        model = Sequential([
            Conv1D(
                filters, kernel_size, strides=1, kernel_regularizer=l2(l2_weight),
                input_shape=(max_words_in_sentence, embedding_dim), padding='valid', activation='relu'),
            MaxPooling1D(5),
            BatchNormalization(),
            Conv1D(
                2 * filters, kernel_size, strides=1, kernel_regularizer=l2(l2_weight),
                input_shape=(max_words_in_sentence, embedding_dim), padding='valid', activation='relu'),
            GlobalMaxPooling1D(),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(1, kernel_regularizer=l2(l2_weight), activation='sigmoid'),
        ])
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model
