from src import middleware
from src.data import FineFoodReviewsDataSet, IMDBDataSet


def apply_middlewares(dataset, middlewares):
    for mw in middlewares:
        dataset = mw(dataset)
    return dataset
