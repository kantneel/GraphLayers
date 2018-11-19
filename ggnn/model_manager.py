class ModelManager(object):
    # Notes

    # in build_graph_model, the model needs to initialize or restore

    def __init__(self, model):
        self.model = model
        self.params = model.params
        self.args = model.params.args

    def build_graph_model(self, mode='training', restore_file=None):
        possible_modes = ['training', 'testing', 'inference']
        if mode not in possible_modes:
            raise NotImplementedError("Mode has to be one of {}".format(", ".join(possible_modes)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model(mode=mode)
            if mode == 'training':
                self.make_train_step()

            if restore_file is None:
                self.initialize_model()
            else:
                self.restore_model(restore_file)

    def initialize_model(self):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.model.sess.run(init_op)

    def restore_model(self, path: str) -> None:
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

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.model.sess.run(variable)

        data_to_save = {
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)
