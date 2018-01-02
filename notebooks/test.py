from random import random

from src.pipeline import CachePipeline, PipelineFunctionStart, ProcessPipelinePart
import pandas as pd


def print_values(*args, **kwargs):
    print("Inside print values", args, kwargs)
    return pd.Series([1, 2, 3])


class TestPipeline(ProcessPipelinePart):

    def process_value(self, data):
        print("inside processing value ", data)
        return data * 3


def cached_value(data):
    print(data)
    return data

pipeline = PipelineFunctionStart(
    method=print_values, pipeline=CachePipeline(
        '/tmp', invalidating_param='test', pipeline=TestPipeline()))
pipeline.get_result(test=slice(5))


