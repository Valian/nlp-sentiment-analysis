from sklearn import pipeline

from shared import transformers
from shared import models
from shared.batch_classifier import KerasBatchClassifier


def get_keras_pipeline(nlp, max_words_in_sentence=200, model=None, **keras_kwargs):
    keras_kwargs.setdefault('epochs', 5)
    keras_kwargs.setdefault('batch_size', 128)
    keras_kwargs.setdefault('validation_split', 0.25)
    
    main_pipeline = [
        ('clear', transformers.ClearTextTransformer()),
        ('nlp_index', transformers.WordsToNlpIndexPipeline(nlp))
    ]
    batch_pipeline = [
        ('nlp_input', transformers.NlpIndexToInputVectorPipeline(nlp, max_words_in_sentence))        
    ]
    
    classifier = KerasBatchClassifier(
        build_fn=models.build_conv1d,
        preprocess_pipeline=pipeline.Pipeline(batch_pipeline), 
        **keras_kwargs)
    
    if model:
        classifier.model = model

    return pipeline.Pipeline(main_pipeline + [('keras', classifier)])