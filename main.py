from framework.model.network import GraphNetwork
from framework.model.layer import GraphLayer
from examples.gated_gnn import GatedLayer
from examples.basic_dense import BasicDenseLayer
from framework.utils.paramspaces import GraphNetworkParams, GraphLayerParams, \
    GraphNetworkPlaceholders, ExperimentParams
from framework.utils.data_processing import DataProcessor
import tensorflow as tf

# first preprocess data to get num edge types, num node types, num classes
train_path = '../Featurization-2/d3_1m.data'
valid_path = '../Featurization-2/d3_100k.data'
batch_size = 10000

layer_params = GraphLayerParams(
    node_embed_size=64,
    node_label_embed_size=64,
    edge_label_embed_size=64)

data_processor = DataProcessor(train_path, batch_size, layer_params, is_training_data=True)
network_params = GraphNetworkParams(
    num_nodes=batch_size,
    num_node_labels=data_processor.num_node_labels,
    num_edge_labels=data_processor.num_edge_labels)

experiment_params = ExperimentParams(
    train=train_path,
    valid=valid_path)

p = data_processor.placeholders
placeholders = GraphNetworkPlaceholders(
    input_node_embeds=p['input_node_embeds'],
    node_labels=p['node_labels'],
    num_graphs=p['num_graphs'],
    graph_nodes_list=p['graph_nodes_list'],
    in_degree_indices=p['in_degree_indices'],
    targets=p['target_values'],
    sorted_messages=p['sorted_messages']
)
net = GraphNetwork(network_params, layer_params, experiment_params, placeholders)
#standard_gated_layer = GatedLayer(layer_params, network_params,
#                                  name='gated_1')

#net.add_layer(standard_gated_layer)
#net.add_layer(standard_gated_layer.clone(name='gated_2'))
#net.add_layer(PoolingLayer(layer_params, network_params))
#net.add_layer(DenseOutputLayer(layer_params, network_params,
#                               [128], data_processor.num_classes))

net.add_layer(GatedLayer(layer_params, network_params))
net.add_layer(GatedLayer(layer_params, network_params))
#net.add_layer(PoolingLayer(layer_params, network_params))
net.run()






