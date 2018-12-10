import sys
sys.path.append('../../')
import pickle
import time
import os
import random
import numpy as np
import tensorflow as tf
from framework.utils.data_processing import DataProcessor
from framework.utils.io import IndexedFileReader, IndexedFileWriter, ThreadedIterator
from framework.utils.paramspaces import GraphNetworkPlaceholders

class GraphNetwork(object):
    def __init__(self, network_params, layer_params, exp_params, placeholders, name='graph_network'):
        self.name = name
        self.network_params = network_params
        self.layer_params = layer_params
        self.exp_params = exp_params
        self.layers = []
        self.ops = {}

    def create_embeddings(self):
        initializer = tf.contrib.layers.xavier_initializer()
        self.node_embeddings = tf.Variable(
            initializer([self.network_params.num_nodes, self.layer_params.node_embed_size]),
            name='node_embeddings')
        self.node_label_embeddings = tf.Variable(
            initializer([self.network_params.num_node_labels, self.layer_params.node_label_embed_size]),
            name='node_label_embeddings')
        self.edge_label_embeddings = tf.Variable(
            initializer([self.network_params.num_edge_labels, self.layer_params.edge_label_embed_size]),
            name='edge_label_embeddings')

    def add_layer(self, new_layer):
        self.layers.append(new_layer)
        # check that layer properties are compatible

    def get_messages(self, current_node_embeds=None):
        """
        Retuns a list of list of tensors
        Each list is for a node which which has neighbors described by node_embed, node_label_embed, edge_label_embed
        Overall shape: [num_nodes, max_degree or num_nodes, node_embed_size + node_label_embed_size + edge_label_embed_size]
        """

        message_sources = []  # list of tensors of message sources of shape [E]
        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_labels = []  # list of tensors of edge type of shape [E]

        all_messages = []
        for edge_label_idx, adj_list_for_edge_label in enumerate(self.placeholders.adjacency_lists):
            edge_sources = adj_list_for_edge_label[:, 0]
            message_node_labels = tf.expand_dims(tf.gather(self.placeholders.node_labels, edge_sources), 1)
            message_edge_labels = tf.expand_dims(tf.ones_like(edge_sources, dtype=tf.int32) * edge_label_idx, 1)

            # [e, 4] (all messages of one particular edge type
            messages_of_edge_label = tf.concat([adj_list_for_edge_label, # 0 - source, 1 - target
                                                message_node_labels, # 2 - source node label
                                                message_edge_labels], axis=1) # 3 - edge label
            all_messages.append(messages_of_edge_label)

        # [m, 4]
        concat_messages = tf.concat(all_messages, axis=0)
        sorted_messages = tf.gather(concat_messages, tf.nn.top_k(-concat_messages[:, 1],
                                                                 k=tf.shape(concat_messages)[0]).indices)

        # all inputs for the layer
        # [m, d2]
        node_label_embeds = tf.one_hot(sorted_messages[:, 2], depth=self.network_params.num_node_labels)
        if current_node_embeds is None:
            current_node_embeds = tf.one_hot(sorted_messages[:, 2], depth=self.layer_params.node_embed_size)
        # [m, d1]

        source_node_embeds = tf.nn.embedding_lookup(current_node_embeds, ids=sorted_messages[:, 0])
        target_node_embeds = tf.nn.embedding_lookup(current_node_embeds, ids=sorted_messages[:, 1])
        # [m, d3]
        edge_label_embeds = tf.one_hot(sorted_messages[:, 3], depth=self.network_params.num_edge_labels)

        # [m, d1+d2+d3]
        sorted_embed_messages = tf.concat([source_node_embeds, node_label_embeds, edge_label_embeds], axis=-1)

        # [n, k, d]
        max_degree = tf.reduce_max(self.placeholders.in_degrees)
        layer_input_messages = tf.scatter_nd(indices=self.placeholders.in_degrees,
                                             updates=sorted_embed_messages,
                                             shape=[tf.reduce_max(sorted_messages[:, 1]) + 1, max_degree,
                                                    self.layer_params.node_embed_size + \
                                                    self.network_params.num_node_labels + \
                                                    self.network_params.num_edge_labels])
        ##################################

        #blank_pad = tf.zeros((self.network_params.num_nodes), dtype=tf.int32)
        #split_pad = tf.zeros((self.network_params.num_nodes - tf.shape(self.placeholders.in_degrees)[0]), dtype=tf.int32)

        #split_vec = blank_pad + tf.concat([self.placeholders.in_degrees, split_pad], axis=0)
        #print("done creating padded split")
        #message_lists = tf.split(sorted_embed_messages, split_vec)
        #print("done splitting messages")

        #max_degree = tf.reduce_max(self.placeholders.in_degrees)
        #padded_message_lists = []
        #for i, m in enumerate(message_lists):
        #    padded_message_lists.append(tf.pad(m, [[0, max_degree - tf.shape(m)[0]], [0, 0]]))
        #    if i % 100 == 0:
        #        print(i)
        ##padded_message_lists = [tf.pad(m, [[0, max_degree - tf.shape(m)[0]], [0, 0]]) for m in message_lists]
        #print("done creating padded message lists")
        #layer_input_messages = tf.stack(padded_message_lists, axis=0)[:tf.shape(self.placeholders.in_degrees)[0]]

        ###################################

        #message_sources = tf.concat(message_sources, axis=0)  # Shape [M]
        #source_node_embeds = tf.nn.embedding_lookup(current_node_embeds, ids=message_sources)
        #source_node_labels = tf.gather(params=self.placeholders.node_labels, indices=message_sources)
        #source_node_label_embeds = tf.nn.embedding_lookup(self.node_label_embeddings, ids=source_node_labels)

        #message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        #target_node_embeds = tf.nn.embedding_lookup(current_node_embeds, ids=message_targets)
        #target_node_labels = tf.gather(params=self.placeholders.node_labels, indices=message_targets)
        #target_node_label_embeds = tf.nn.embedding_lookup(self.node_label_embeddings, ids=target_node_labels)

        #message_edge_labels = tf.concat(message_edge_labels, axis=0)  # Shape [M]
        #edge_label_embeds = tf.nn.embedding_lookup(self.edge_label_embeddings, ids=message_edge_labels)

        # This is the input to the layer. Some notes:
        # 1. I think it also makes sense to
        #    include the target node embed and target node label embed
        #    since that could be used for recurrent layers, but this does not currently do so.
        # 2. Some layers require embeddings, but other ones require the label index itself.
        #    An example is when there are separate weights/parameters for every label,
        #    which is the case for GGNN.
        #concat_embeds = tf.concat([source_node_embeds, source_node_label_embeds, edge_label_embeds], axis=1)
        #return (concat_embeds, message_targets)
        return layer_input_messages

    def make_model(self):
        """Create the tensorflow graph that encodes the network"""
        self.placeholders = self.define_placeholders()

        current_node_embeds = None
        for layer in self.layers:
            #messages_for_nodes = self.get_messages(current_node_embeds)
            #output_embeds = []
            #for message_list in messages_for_nodes:
            #    # computing new node embedding based off of incoming messages
            #    output_embeds.append(layer(message_list))
            ## updated embeddings for all nodes
            #current_node_embeds = tf.stack(output_embeds)

            layer.create_weights()
            num_timesteps = getattr(layer, 'num_timesteps', 1)
            for i in range(num_timesteps):
                layer_input_messages = self.get_messages(current_node_embeds)
                current_node_embeds = layer(layer_input_messages)

        # Pooling the nodes for each graph. I suppose customizing
        # this will be a feature of the GraphNetwork class.
        logits = tf.unsorted_segment_sum(
            data=current_node_embeds,
            segment_ids=self.placeholders.graph_nodes_list,
            num_segments=self.placeholders.num_graphs)
        labels = self.placeholders.targets
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)

        self.ops['loss'] = tf.reduce_mean(loss)
        probabilities = tf.nn.softmax(logits)

        correct_prediction = tf.equal(tf.argmax(probabilities, -1, output_type=tf.int32), self.placeholders.targets)
        self.ops['accuracy_task'] = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        top_k = tf.nn.top_k(probabilities, self.exp_params.top_k)
        self.ops['preds'] = top_k.indices
        self.ops['probs'] = top_k.values

    def define_placeholders(self):
        placeholders = GraphNetworkPlaceholders(
            input_node_embeds=tf.placeholder(
                tf.float32, [None, self.layer_params.node_embed_size], name='input_node_embeds'),
            node_labels=tf.placeholder(
                tf.int32, [None], name='node_labels'),
            adjacency_lists=tuple([tf.placeholder(
                tf.int32, [None, 2], name='adjacency_e%s' % e) for e in range(self.network_params.num_edge_labels)]),
            num_graphs=tf.placeholder(tf.int32, [], name='num_graphs'),
            graph_nodes_list=tf.placeholder(tf.int32, [None], name='graph_nodes_list'),
            targets=tf.placeholder(tf.int32, [None], name='targets'),
            in_degrees=tf.placeholder(tf.int32, [None, 2], name='in_degrees')
        )
        return placeholders

    def build_graph_model(self, mode='training', restore_file=None):
        possible_modes = ['training', 'testing', 'inference']
        if mode not in possible_modes:
            raise NotImplementedError("Mode has to be one of {}".format(", ".join(possible_modes)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        #from tensorflow.python import debug as tf_debug
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        with self.graph.as_default():
            self.create_embeddings()
            self.make_model()
            if mode == 'training':
                self.make_train_step()

            if restore_file is None:
                self.initialize_model()
            else:
                self.restore_model(restore_file)

    def initialize_model(self):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path):
        debug = self.args.get('debug', False)
        if debug:
            print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    if debug:
                        print("Restoring weights for %s" % variable.name)
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    if debug:
                        print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)

            if debug:
                for var_name in data_to_load['weights']:
                    if var_name not in used_vars:
                        print('Saved weights for %s not used by model.' % var_name)

            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.model.sess.run(restore_ops)

    def save_model(self, path):
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #if False:
        #    graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
        #    filtered_vars = []
        #    for var in trainable_vars:
        #        if var not in graph_vars:
        #            filtered_vars.append(var)
        #        else:
        #            print("Freezing weights of variable %s." % var.name)
        #    trainable_vars = filtered_vars

        optimizer = tf.train.AdamOptimizer(self.exp_params.lr)
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.exp_params.clamp_grad_norm), var))
            else:
                clipped_grads.append((grad, var))

        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)

        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            #if self.exp_params.get('restore', None) is not None:
            #    _, valid_acc, _ = self.run_train_epoch("Resumed (validation)", valid_data, False)
            #    best_val_acc = valid_acc
            #    best_val_acc_epoch = 0
            #    print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            #else:
            (best_val_acc, best_val_acc_epoch) = (0.0, 0)

            # this will be undone in the first epoch
            for epoch in range(1, self.exp_params.num_epochs + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_acc, train_speed = self.run_train_epoch(
                    "epoch %i (training)" % epoch, self.train_data_processor)
                accs_str = "%.5f" % train_acc
                print("\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f" % (train_loss,
                                                                                      accs_str,
                                                                                      train_speed))

                valid_loss, valid_acc, valid_speed = self.run_train_epoch(
                    "epoch %i (validation)" % epoch, self.valid_data_processor)
                accs_str = "%.5f" % valid_acc
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f" % (valid_loss,
                                                                                      accs_str,
                                                                                      valid_speed))

                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_acc, train_speed),
                    'valid_results': (valid_loss, valid_acc, valid_speed),
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)

                if valid_acc > best_val_acc:
                    self.save_model(self.best_model_file)
                    print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (
                        valid_acc, best_val_acc, self.best_model_file))
                    best_val_acc = valid_acc
                    best_val_acc_epoch = epoch

                elif epoch - best_val_acc_epoch >= self.exp_params.patience:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % \
                          self.exp_params.patience)
                    break

    def run_train_epoch(self, epoch_name, data_processor):
        loss = 0
        accuracy = 0
        accuracy_op = self.ops['accuracy_task']
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(data_processor.make_minibatch_iterator(), max_queue_size=50)
        for step, batch_data_dict in enumerate(batch_iterator):
            batch_data = {
                self.placeholders.input_node_embeds : batch_data_dict['input_node_embeds'],
                self.placeholders.node_labels : batch_data_dict['node_labels'],
                self.placeholders.graph_nodes_list : batch_data_dict['graph_nodes_list'],
                self.placeholders.targets : batch_data_dict['targets'],
                self.placeholders.num_graphs : batch_data_dict['num_graphs'],
                self.placeholders.adjacency_lists : batch_data_dict['adjacency_lists'],
                self.placeholders.in_degrees : batch_data_dict['in_degrees']
            }
            num_graphs = batch_data[self.placeholders.num_graphs]
            processed_graphs += num_graphs
            if data_processor.is_training_data:
                #batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                #    'out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_op, self.ops['train_step']]
            else:
                #batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_op]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracy) = (result[0], result[1])
            loss += batch_loss * num_graphs
            accuracy += batch_accuracy * num_graphs

            print("Running %s, batch %i (has %i graphs). "
                  "Loss so far: %.4f. Accuracy so far: %.4f" % (epoch_name,
                                                                step,
                                                                num_graphs,
                                                                loss / processed_graphs,
                                                                accuracy / processed_graphs),
                  end='\r')

        accuracy = accuracy / processed_graphs
        loss = loss / processed_graphs
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracy, instance_per_sec

    def run(self):
        if self.exp_params.mode == 'train':
            self.run_training()

    def run_training(self):
        self.wdir = self.exp_params.wdir
        if self.wdir is None:
            self.wdir = "_".join(["run", time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
            os.system('rm -rf {}'.format(self.wdir))
            os.system('mkdir -p {}'.format(self.wdir))

        if not os.path.exists(self.wdir):
            os.system('mkdir -p {}'.format(self.wdir))

        #  Save away the params
        with open('{}/params.pkl'.format(self.wdir), 'wb') as f:
            pickle.dump(self.exp_params, f)

        #  For readability
        with open('{}/params.txt'.format(self.wdir), 'w') as f:
            print(str(self.exp_params), file=f)

        self.log_file = '{}/log.json'.format(self.wdir)
        self.best_model_file = '{}/model_best.pickle'.format(self.wdir)

        random.seed(self.exp_params.rng)
        np.random.seed(self.exp_params.rng)

        #  Setup the model
        self.build_graph_model(mode='training', restore_file=None)

        #  Load up the data
        self.train_data_processor = DataProcessor(self.exp_params.train, self.network_params.num_nodes,
                                        self.layers[0].layer_params, is_training_data=True)

        self.valid_data_processor = DataProcessor(self.exp_params.valid, self.network_params.num_nodes,
                                        self.layers[0].layer_params, is_training_data=False)

        self.train()

