def preprocess_data(path, tie_fwd_bkwd=True):
    reader = IndexedFileReader(path)

    num_fwd_edge_types = 0
    annotation_size = 0
    num_classes = 0
    depth = 1
    for g in tqdm.tqdm(reader, desc='Preliminary Data Pass', dynamic_ncols=True):
        num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['edges']] + [-1]) + 1)
        annotation_size = max(annotation_size, max(g['node_features']) + 1)
        num_classes = max(num_classes, g['label'] + 1)

    params = {}
    params['num_edge_types'] = num_fwd_edge_types * (1 if tie_fwd_bkwd else 2)
    params['annotation_size'] = annotation_size
    params['num_classes'] = num_classes
    reader.close()

    return params

def load_data(path, use_memory=False, use_disk=False, is_training_data=False):
    reader = IndexedFileReader(path)
    if use_memory
        result = process_raw_graphs(reader, is_training_data)
        reader.close()
        return result

    if use_disk:
        w = IndexedFileWriter(path + '.processed')
        for d in tqdm.tqdm(reader, desc='Dumping processed graphs to disk'):
            w.append(pickle.dumps(process_raw_graph(d)))

        w.close()
        reader.close()
        return IndexedFileReader(path + '.processed')

    #  We won't pre-process anything. We'll convert on-the-fly. Saves memory but is very slow and wasteful
    reader.set_loader(lambda x: process_raw_graph(pickle.load(x)))
    return reader

def process_raw_graphs(raw_data, annotation_size, is_training_data=False):
    processed_graphs = []
    for d in tqdm.tqdm(raw_data, desc='Processing Raw Data'):
        processed_graphs.append(process_raw_graph(d, annotation_size))

    if is_training_data:
        np.random.shuffle(processed_graphs)

    return processed_graphs

def process_raw_graph(graph, annotation_size):
    (adjacency_lists, num_incoming_edge_per_type) = graph_to_adjacency_lists(graph['edges'])
    return {"adjacency_lists": adjacency_lists,
            "num_incoming_edge_per_type": num_incoming_edge_per_type,
            "init": to_one_hot(graph["node_features"], annotation_size),
            "label": graph.get("label", 0)}

def graph_to_adjacency_lists(graph, num_edge_labels, tie_fwd_bkwd=True) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    adj_lists = collections.defaultdict(list)
    num_incoming_edges_dicts_per_type = {}
    for src, e, dest in graph:
        fwd_edge_type = e
        adj_lists[fwd_edge_type].append((src, dest))
        if fwd_edge_type not in num_incoming_edges_dicts_per_type:
            num_incoming_edges_dicts_per_type[fwd_edge_type] = collections.defaultdict(int)

        num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
        if tie_fwd_bkwd:
            adj_lists[fwd_edge_type].append((dest, src))
            num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

    final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                       for e, lm in adj_lists.items()}

    # Add backward edges as an additional edge type that goes backwards:
    if not (tie_fwd_bkwd):
        for (edge_type, edges) in adj_lists.items():
            bwd_edge_type = num_edge_labels + edge_type
            final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
            if bwd_edge_type not in num_incoming_edges_dicts_per_type:
                num_incoming_edges_dicts_per_type[bwd_edge_type] = collections.defaultdict(int)

            for (x, y) in edges:
                num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

    return final_adj_lists, num_incoming_edges_dicts_per_type

def to_one_hot(vals: List[int], depth: int):
    res = []
    for val in vals:
        v = [0] * depth
        v[val] = 1
        res.append(v)

    return res
