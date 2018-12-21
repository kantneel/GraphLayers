from framework.model.network import GraphNetwork
from framework.model.layer import GraphLayer
from examples.gated_gnn import GatedLayer
from examples.basic_dense import BasicDenseLayer
from framework.utils.paramspaces import GraphNetworkParams, GraphLayerParams, \
    GraphNetworkPlaceholders, ExperimentParams
from framework.utils.data_processing import DataProcessor
import argparse
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for GatedLayer Demo')
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--wdir', type=str, default=None)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--clamp-grad-norm', type=float, default=1.)
    parser.add_argument('--rng', type=int, default=2018)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--use-sparse', action='store_true',
                        help='whether to use sparse tensors')
    parser.add_argument('--restore', type=str, default=None,
                        help='path to model to restore')
    parser.add_argument('--train-path', type=str, default=None,
                        help='relative path to training dataset created with IndexedFileWriter')
    parser.add_argument('--valid-path', type=str, default=None,
                        help='relative path to validation dataset created with IndexedFileWriter')
    parser.add_argument('--node-embed-size', type=int, default=64)
    parser.add_argument('--node-label-embed-size', type=int, default=64)
    parser.add_argument('--edge-label-embed-size', type=int, default=64)
    args = parser.parse_args()

    # first preprocess data to get num edge types, num node types, num classes
    train_data_processor = DataProcessor(args.train_path, args.batch_size, args.node_embed_size, is_training_data=True)
    valid_data_processor = DataProcessor.copy_params(train_data_processor, args.valid_path, is_training_data=False)

    # define parameters primarily used by layers
    layer_params = GraphLayerParams(
        node_embed_size=args.node_embed_size,
        node_label_embed_size=args.node_label_embed_size,
        edge_label_embed_size=args.edge_label_embed_size)

    # define parameters used by both networks as well as the layers in them
    network_params = GraphNetworkParams(
        num_nodes=args.batch_size,
        num_node_labels=train_data_processor.num_node_labels,
        num_edge_labels=train_data_processor.num_edge_labels,
        use_sparse=args.use_sparse)

    # define parameters that are used for governing the experiment
    experiment_params = ExperimentParams(
        num_epochs=args.num_epochs, lr=args.learning_rate,
        patience=args.patience, wdir=args.wdir, top_k=args.top_k,
        clamp_grad_norm=args.clamp_grad_norm, rng=args.rng,
        restore=args.restore)

    net = GraphNetwork(network_params, layer_params, experiment_params,
                       train_data_processor=train_data_processor,
                       valid_data_processor=valid_data_processor)

    net.add_layer(GatedLayer(layer_params, network_params, name='first'))
    net.add_layer(GatedLayer(layer_params, network_params, name='second'))
    net.run_training()

