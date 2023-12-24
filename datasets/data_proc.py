import logging

import torch

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.preprocessing import StandardScaler

#shraban
from scipy import sparse
import os
import numpy as np


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
    "block" : "/home/shraban/Paper3/KPGNN/KPGNN/incremental_test_100messagesperday/"
}
#shraban
def load_small_dataset(dataset_name):
    assert dataset_name.split('_')[0] in GRAPH_DICT, f"Unknow dataset: {dataset_name}." #shraban
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    elif dataset_name.startswith("cor") or dataset_name.startswith("pub") or dataset_name.startswith("cit"): #shraban
        dataset = GRAPH_DICT[dataset_name]()
    #shraban
    elif:
        data_path = GRAPH_DICT[dataset_name.split('_')[0]]

    
    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    
    #shraban
    elif dataset_name.startswith('block'):
        block_num = dataset_name.split('_')[1]
        i,j = np.nonzero(sparse.load_npz('/home/shraban/Paper3/KPGNN/KPGNN/incremental_test_100messagesperday/'+block_num+'/s_bool_A_tid_tid.npz').toarray()) # load from file
        edge_index = torch.tensor([i.tolist(), j.tolist()]).to(int)
        feats = torch.from_numpy(np.load('/home/shraban/Paper3/KPGNN/KPGNN/incremental_test_100messagesperday/'+block_num+'/features.npy')) # load from file
        label = torch.from_numpy(np.load('/home/shraban/Paper3/KPGNN/KPGNN/incremental_test_100messagesperday/'+block_num+'/labels.npy')).to(int) # load from file
        # labels are not always sequential so we need to make them sequential
        label_map = {i.item() : j.item() for i,j in zip(torch.unique(label), torch.arange(len(torch.unique(label))).to(int))}
        label = torch.tensor([label_map[i.item()] for i in label]).to(int)
        
        train_split, valid_split, test_split = torch.split(torch.arange(feats.size(0))[torch.randperm(feats.size(0))],[int(0.1*feats.size(0)),int(0.1*feats.size(0)),feats.size(0)-(2*int(0.1*feats.size(0)))],dim=0)
        graph = dgl.DGLGraph((edge_index[0], edge_index[1]))
        graph.ndata['feat'] = feats
        graph.ndata['label'] = label
        graph.ndata['test_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool).scatter_(0, test_split, True)
        graph.ndata['val_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool).scatter_(0, valid_split, True)
        graph.ndata['train_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool).scatter_(0, train_split, True)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    
    num_features = graph.ndata["feat"].shape[1]
    
    #shraban
    if dataset_name.startswith('block'):
        num_classes = torch.unique(graph.ndata["label"]).shape[0]
    else:
        num_classes = dataset.num_classes
    
    return graph, (num_features, num_classes)

def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


def scale_feats(x):
    logging.info("### scaling features ###")
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
