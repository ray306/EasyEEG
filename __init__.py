from .default import *

from . import io
from . import group
from . import structure

from . import graph
from . import statistics
from . import algorithms

import importlib
def reload(module=None):
    for _ in range(2):
        importlib.reload(io.save)
        importlib.reload(io.load)
        importlib.reload(io)
        importlib.reload(structure)
        importlib.reload(structure.group)
        importlib.reload(group)
        importlib.reload(algorithms)
        importlib.reload(algorithms.basic)
        importlib.reload(algorithms.stats_methods) 
        importlib.reload(graph)
        importlib.reload(graph.figure_group) 
        importlib.reload(graph.figure_unit)
        if module != None:
            importlib.reload(module)

print('EasyEEG loaded.')
