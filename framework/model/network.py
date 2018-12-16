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

    def add_layer(self, new_layer):
        self.layers.append(new_layer)
        # check that layer properties are compatible

    def get_filler_tensor(self, input_split_type, max_degree,
                          mask_indices, mask_update_shape):
        net_p = self.network_params
        num_nodes = tf.size(self.placeholders.node_labels)
        split_to_shape = {
            'nodes'       : [num_nodes,
                             max_degree, 1],
            'node_labels' : [net_p.num_node_labels,
                             num_nodes,
                             max_degree, 1],
            'edge_labels' : [net_p.num_edge_labels,
                             num_nodes,
                             max_degree, 1]
        }
        if input_split_type not in split_to_shape.keys():
            raise Exception("arg input_split_type must be one of 'nodes', \
                            'node_labels' or 'edge_labels'")

        filler_tensor_shape = split_to_shape[input_split_type]
        fill_values = [num_nodes, net_p.num_node_labels, net_p.num_edge_labels]
        blank_tensors = []
        for val in fill_values:
            blank_tensors.append(tf.fill(filler_tensor_shape, val))

        filler_tensor = tf.concat(blank_tensors, axis=-1)

        # modify this shape for use in mask creation
        mask_tensor_shape = filler_tensor_shape[:-1] + [3]
        filler_mask = tf.scatter_nd(
            indices=mask_indices,
            updates=tf.ones(mask_update_shape),
            shape=mask_tensor_shape)

        # now filler_tensor is zero where updates will be applied
        # and equal to num_nodes/num_node_labels/num_edge_labels elsewhere
        filler_tensor = filler_tensor * tf.cast(tf.equal(filler_mask, 0), tf.int32)
        return filler_tensor

    def get_messages(self, layer):
        """
        Takes in a layer and returns the types of messages that the layer
        needs for input based on the layer's InputConfig
        """
        sorted_messages = self.placeholders.sorted_messages
        # Indices [0, 2, 3] correspond to source, source label and edge label ids
        sorted_embed_messages = tf.transpose(tf.gather(
            tf.transpose(sorted_messages), [0, 2, 3]))
        #sorted_embed_messages = sorted_messages[:, [0, 2, 3]]
        max_degree = tf.reduce_max(self.placeholders.in_degrees[:, 1])
        # scatter into [n, k, 3] where we have [n, k, 1] * num_nodes,
        # scatter options
        argv = []
        config = layer.get_input_config()
        if config.source_only:
            # just split by source node
            # shape: [n, k, 3]
            filler_tensor = self.get_filler_tensor(
                input_split_type='nodes',
                max_degree=max_degree,
                mask_indices=self.placeholders.in_degrees,
                mask_update_shape=tf.shape(sorted_embed_messages))

            messages_by_source_only = tf.scatter_nd(
                indices=self.placeholders.in_degrees,
                updates=sorted_embed_messages,
                shape=tf.shape(filler_tensor))
            # now messages_by_source_tensor has ids in correct positions
            # and ids which correspond to zero embeddings elsewhere
            messages_by_source_only += filler_tensor
            argv.append(messages_by_source_only)

        if config.source_node_labels:
            # also split by source node label
            # shape: [num_node_labels, n, k, 3]
            source_node_labels = tf.expand_dims(sorted_messages[:, 2], axis=1)
            split_by_label_indices = tf.concat(
                [source_node_labels, self.placeholders.in_degrees], axis=1)

            filler_tensor = self.get_filler_tensor(
                input_split_type='node_labels',
                max_degree=max_degree,
                mask_indices=split_by_label_indices,
                mask_update_shape=tf.shape(sorted_embed_messages))

            messages_by_source_label = tf.scatter_nd(
                indices=split_by_label_indices,
                updates=sorted_embed_messages,
                shape=tf.shape(filler_tensor))

            messages_by_source_label += filler_tensor
            argv.append(messages_by_source_label)

        if config.edge_labels:
            # also split by edge label
            # shape: [num_edge_labels, n, k, 3]
            edge_labels = tf.expand_dims(sorted_messages[:, 3], axis=1)
            split_by_label_indices = tf.concat(
                [edge_labels, self.placeholders.in_degrees], axis=1)

            filler_tensor = self.get_filler_tensor(
                input_split_type='edge_labels',
                max_degree=max_degree,
                mask_indices=split_by_label_indices,
                mask_update_shape=tf.shape(sorted_embed_messages))

            messages_by_edge_label = tf.scatter_nd(
                indices=split_by_label_indices,
                updates=sorted_embed_messages,
                shape=tf.shape(filler_tensor))

            messages_by_edge_label += filler_tensor
            argv.append(messages_by_edge_label)

        return argv

    def make_model(self):
        """Create the tensorflow graph that encodes the network"""
        self.placeholders = self.define_placeholders()
        current_node_embeds = tf.one_hot(self.placeholders.node_labels,
                                         depth=self.layer_params.node_embed_size)
        self.placeholders.in_degrees = tf.squeeze(self.placeholders.in_degrees, 0)
        for layer in self.layers:
            layer.create_weights()
            num_timesteps = getattr(layer, 'num_timesteps', 1)
            for i in range(num_timesteps):
                layer_input_args = self.get_messages(layer)
                layer_input_args.append(current_node_embeds)
                current_node_embeds = layer(*layer_input_args)

        current_node_embeds = tf.layers.dense(current_node_embeds,
                                              self.train_data_processor.num_classes)
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
            in_degrees=tf.placeholder(tf.int32, [None, 2], name='in_degrees'),
            sorted_messages=tf.placeholder(tf.int32, [None, 4], name='sorted_messages')
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
                self.placeholders.in_degrees : batch_data_dict['in_degrees'],
                self.placeholders.sorted_messages : batch_data_dict['sorted_messages']
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


        #  Load up the data
        self.train_data_processor = DataProcessor(self.exp_params.train, self.network_params.num_nodes,
                                        self.layers[0].layer_params, is_training_data=True)
        self.valid_data_processor = DataProcessor(self.exp_params.valid, self.network_params.num_nodes,
                                        self.layers[0].layer_params, is_training_data=False)
        #  Setup the model
        self.build_graph_model(mode='training', restore_file=None)

        self.train()
