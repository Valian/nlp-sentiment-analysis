from keras.models import load_model


class MachineLearningModel(object):

    def __init__(self, model):
        self.model = model

    def predict(self, input):
        raise NotImplementedError()

    def train(self, train_dataset, validation_dataset, **kwargs):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    @classmethod
    def load(cls, filename):
        raise NotImplementedError()


class KerasModel(MachineLearningModel):

    def save(self, filepath):
        self.model.save(filepath)

    def predict(self, input):
        pass

    @classmethod
    def load(cls, filepath):
        return cls(load_model(filepath))

    def train(self, train_dataset, validation_dataset, **kwargs):
        n_train_batches, train_generator = train_dataset.to_generator(batch_size=128)
        n_test_batches, test_generator = validation_dataset.to_generator(batch_size=128)
        return self.model.fit_generator(
            train_generator,
            steps_per_epoch=n_train_batches,
            epochs=kwargs.get('epochs', 5),
            verbose=1,
            validation_data=test_generator,
            validation_steps=n_test_batches)
