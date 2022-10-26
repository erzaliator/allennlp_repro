import math
from typing import Tuple, Dict, List, Any, Union

from allennlp.common import Registrable

class TransformationFunction(Registrable):
    def __call__(self, xs, tokenwise):
        raise NotImplementedError("Please use a class that inherits from TransformationFunction")

@TransformationFunction.register("bins")
class Bins(TransformationFunction):
    def __init__(self, bins: List[Tuple[Union[int, float], Union[int, float]]]):
        # bins is a list of ints such that any value falls into it iff `bin[0] <= x < bin[1]`
        self.bins = bins

    def _bin(self, x):
        for i, (start, end) in enumerate(self.bins):
            if start <= x < end:
                return str(i)
        raise Exception(f"No bin found for value {x}! Bins: {self.bins}")

    def __call__(self, xs, tokenwise):
        if tokenwise:
            return [self._bin(x) for x in xs]
        else:
            return self._bin(xs)


distance = Bins([[-1e9, -8], [-8, -2], [-2, 0], [0, 2], [2, 8], [8, 1e9]])
print(distance.__call__(2, tokenwise=False))