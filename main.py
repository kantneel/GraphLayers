from framework.model.base_network import GraphNetwork
from framework.model.layer import GraphLayer
from examples.gated_gnn import GatedLayer
from examples.graph_pool import PoolingLayer
from examples.output_layers import DenseOutputLayer
from framework.utils.paramspaces import GraphNetworkParams, GraphLayerParams, \
    GraphNetworkPlaceholders, ExperimentParams
from framework.utils.data_processing import DataProcessor

# first preprocess data to get num edge types, num node types, num classes
train_path = 'depth1_data/d3_100k.data'
valid_path = train_path
batch_size = 8192

layer_params = GraphLayerParams(
    node_embed_size=256,
    node_label_embed_size=64,
    edge_label_embed_size=64)

data_processor = DataProcessor('depth1_data/d3_100k.data', batch_size, layer_params, is_training_data=True)
network_params = GraphNetworkParams(
    num_nodes=batch_size,
    num_node_labels=data_processor.num_node_labels,
    num_edge_labels=data_processor.num_edge_labels)

experiment_params = ExperimentParams(
    train=train_path,
    valid=valid_path)

p = data_processor.placeholders
placeholders = GraphNetworkPlaceholders(
    input_node_embeds=p['initial_node_representation'],
    adjacency_lists=p['adjacency_lists'],
    num_incoming_edges_per_label=p['num_incoming_edges_per_label'],
    num_graphs=p['num_graphs'],
    graph_nodes_list=p['graph_nodes_list'],
    targets=p['target_values']
)

net = GraphNetwork(network_params, experiment_params, placeholders)
standard_gated_layer = GatedLayer(layer_params, network_params,
                                  name='gated_1')

net.add_layer(standard_gated_layer)
net.add_layer(standard_gated_layer.clone(name='gated_2'))
net.add_layer(PoolingLayer(layer_params, network_params))
net.add_layer(DenseOutputLayer(layer_params, network_params,
                               [128], data_processor.num_classes))

net.run()






