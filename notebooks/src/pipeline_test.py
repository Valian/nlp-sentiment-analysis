from src.pipeline import ProcessPipelinePart, PipelineStart, PipelineBase, SplitPipeline


class Multiply(ProcessPipelinePart):

    def __init__(self, multiplier):
        self.multiplier = multiplier

    def process(self, data):
        return data * self.multiplier


class ProvidedDataPipeline(PipelineStart):

    def __init__(self, initial_data, pipeline: PipelineBase=None):
        super().__init__(pipeline)
        self.initial_data = initial_data

    def get_initial_data(self, *args, **kwargs):
        return self.initial_data


def test_pipeline_start():
    calculator = ProvidedDataPipeline(5)
    result = calculator.get_result()
    assert result == 5


def test_basic_pipeline():
    calculator = ProvidedDataPipeline(5, pipeline=Multiply(2))
    result = calculator.get_result()
    assert result == 10


def test_split_pipeline():
    result = ProvidedDataPipeline({'a': 1, 'b': 10}, pipeline=SplitPipeline({
        'a': Multiply(5),
        'b': Multiply(0.5)
    })).get_result()
    assert result == {'a': 5, 'b': 5}
