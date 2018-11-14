from layer import GraphLayer

class GraphLayerList(object):
    """
    A GraphLayerList is an iterable module that holds GraphLayers in a
    sequence and calls each of them in order when called.
    """
    def __init__(self, layer_list=[]):
        """
        Args

        layer_list: list in which each object is of type GraphLayer
        """

        self.layer_list = layer_list

        self._check_valid_args()

    def _check_valid_args(self):
        """A bunch of assertions to make sure types make sense"""
        pass

    def _check_compatible_append(self, layer):
        """
        Check properties of new layer to make sure it is compatible
        with the last layer currently in the list.

        - check number of output nodes in current last layer and number
            of input nodes in this new layer
        """
        compatible_append = True

        # run some checks
        return compatible_append

    def append(self, layer):
        try:
            assert isinstance(layer, GraphLayer)
        except AssertionError:
            raise Exception("Invalid Layer")

        if self._check_compatible_append(layer):
            self.layer_list.append(layer)

    def __iter__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass


