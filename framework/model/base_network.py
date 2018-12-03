import sys
sys.path.append('../../')
import pickle
import time
import os
import random
import numpy as np
import tensorflow as tf
from framework.utils.io import IndexedFileReader, IndexedFileWriter, ThreadedIterator

class GraphNetwork(object):
    def __init__(self, network_params, exp_params, placeholders):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.network_params = network_params
            self.exp_params = exp_params
            self.placeholders = placeholders
            self.layers = []
            self.ops = {}

    def add_layer(self, new_layer):
        self.layers.append(new_layer)
        # check that layer properties are compatible

    def make_model(self, placeholders):
        outputs = [placeholders]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        results = outputs[-1]

        logits = results.input_node_embeds
        labels = placeholders.targets
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)

        self.ops['loss'] = tf.reduce_mean(loss)
        probabilities = tf.nn.softmax(logits)

        correct_prediction = tf.equal(tf.argmax(probabilities, -1), placeholders.targets)
        self.ops['accuracy_task'] = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        top_k = tf.nn.top_k(probabilities, self.exp_params.top_k)
        self.ops['preds'] = top_k.indices
        self.ops['probs'] = top_k.values

    def build_graph_model(self, mode='training', restore_file=None):
        possible_modes = ['training', 'testing', 'inference']
        if mode not in possible_modes:
            raise NotImplementedError("Mode has to be one of {}".format(", ".join(possible_modes)))

        self.make_model(self.placeholders)
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
        print(trainable_vars)
        #if self.params.args.freeze_graph_model:
        if False:
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars

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
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders.num_graphs]
            processed_graphs += num_graphs
            if is_training:
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
        self.train_data_processor = DataProcessor(self.exp_params.train, self.network_params,
                                        self.layers[0].layer_params, is_training_data=True)

        self.valid_data_processor = DataProcessor(self.exp_params.valid, self.network_params,
                                        self.layer[0].layer_params, is_training_data=False)

        self.train()

