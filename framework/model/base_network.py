import sys
sys.path.append('../../')
from framework.utils.io import IndexedFileReader, IndexedFileWriter

class GraphNetwork(object):
    def __init__(self, network_params):
        self.network_params = network_params
        self.layers = []

    def add_layer(self, new_layer):
        self.layers.append(new_layer)
        # check that layer properties are compatible

    def __call__(self, placeholders):
        results = placeholders
        for layer in self.layers:
            results = layer(results)

        return results
