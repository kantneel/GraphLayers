# GraphLayers

### Requirements
- `python` v3.6
- `tensorflow` v1.12
- `tqdm`

### Gated Graph Neural Network Demo
The main demonstration I've put together is that of a Gated Graph Neural Network (GGNN) based off [Li et. al. (2015)](https://arxiv.org/abs/1511.05493). Much of the backend for data processing and running experiments has been borrowed from [the Microsoft Research implementation](https://github.com/Microsoft/gated-graph-neural-network-samples). To run a demo, first get the data (will figure this out soon). Then, you can run the example with 
```
python3 main.py --train-path <relative path> --valid-path <relative-path> (--use-sparse)
```
`--use-sparse` does a lot of the processing using sparse tensors and ops in an effort to be more memory efficient. Unfortunately, it also runs much more slowly because it is hard to effectively leverage GPU parallelism with sparse ops. 

### What the Framework Does and How to Use It
This framework defines two central classes: `GraphNetwork` (network) and `GraphLayer` (layer). The network is responsible for backend processing of batches and assembly of input tensors which are fed into a layer. The layer processes the input tensors and produces a new set of node embeddings. This two-step process continues until a final set of node embeddings are computed, at which point the network accumulates them to get graph embeddings and a typical classification and learning procedure follows. 

(I'll explain this more later.)

In general, you can define a layer by subclassing `GraphLayer`. For this class there are two abstract methods, `create_node_label_embeddings()` and `create_edge_label_embeddings()`, because it is a layer's responsibility to define these. I have created default functions that can be easily used (e.g. anything in the examples folder). A layer also has an `InputConfig`, which specifies the layout of its input message tensor. The options are characterized as "source_only", "source_node_labels", "edge_labels", which indicate an extra dimension in the input message tensor, making it \[E, N, K, 3\] rather than \[N, K, 3\], where N is the number of nodes in the batch and K is the maximum in-degree of those nodes. A layer will also take in the current set of node embeddings, because they may be necessary for recurrent layers like that of the GGNN. To see the precise mechanics of how inputs are passed into layers successively, see the `make_model()` method of `GraphNetwork`. A large part of the work is done in the `get_messages()` model which takes into account whether the network is using sparse computation and also the InputConfig of the specific layers. 

### Why A Framework for Graph Neural Networks?
Graph Neural Networks (GNNs) are a less commonly known family of neural networks. As the name suggests, they operate on graphs. This makes GNNs the most general neural network family in deep learning today, since the domains of CNNs (typically images/video) and RNNs (sequences) can be modeled as (directed) graphs. There are also many other common objects of interest which are naturally encoded as graphs, such as molecules, social networks and abstract knowledge. In coming years, the use of GNNs will likely become more prevalent as researchers explore the promise of deep learning in these domains as well. 

Compared to other mainstream deep learning architectures, there are fewer resources for understanding the fundamental mechanisms that underpin how GNNs work. But the gist is that GNNs learn functions of graphs to their labels in a very similar way to other deep learning models. They learn embeddings of the nodes, which in turn can be reduced to an embedding of the entire graph, which can be transformed into predictions about its label. The embeddings of nodes are fundamentally dictated by their neighbors. 

(I'll summarize this more later.)
