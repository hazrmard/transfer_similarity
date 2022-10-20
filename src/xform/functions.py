from typing import Callable

import numpy as np



class InvertibleFunction:
    """Use as InvertibleFunction() @ and InvertibleFunction.invert() @"""
    def __init__(self, func: Callable, inverse: Callable):
        self.func, self.inverse = func, inverse
    def __call__(self, *args):
        return self.func(*args)
    def __matmul__(self, *args):
        return self(*args)
    def invert(self):
        return InvertibleFunction(self.inverse, self.func)