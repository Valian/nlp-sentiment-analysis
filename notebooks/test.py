from src.data import FineFoodReviewsDataSet
from src.middleware import BinarizeLabelsMiddleware, ClearTextMiddleware, WordsToNlpIndexMiddleware, \
    NlpIndexToInputVectorMiddleware, CacheMiddleware
import spacy

print("Loading spacy...")



data_stream = dataset.to_generator(batch_size=3)
for i, (data, labels) in zip(range(2), data_stream):
    print(data)


print(dataset.get_settings_hash().hexdigest())
