from framework.model.base_network import GraphNetwork
from framework.model.layer import GraphLayer
from examples.gated_gnn import GatedLayer
from framework.utils.paramspaces import GraphNetworkParams, GraphLayerParams, GraphNetworkPlaceholders
from framework.utils.data_processing import DataProcessor

# first preprocess data to get num edge types, num node types, num classes

data_processor = DataProcessor(' ', is_training_data=True)
train_data = data_processor.load_data(use_memory=True)

network_params = GraphNetworkParams(
    num_nodes=8192,
    num_node_labels=data_processor.num_node_labels,
    num_edge_labels=data_processor.num_edge_labels)

layer_params = GraphLayerParams(
    node_embed_size=128,
    node_label_embed_size=32,
    edge_label_embed_size=32)

net = GraphNetwork(network_params)

