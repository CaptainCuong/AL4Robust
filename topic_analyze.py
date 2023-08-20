from top2vec import Top2Vec
from utils import *
from contextlib import contextmanager
import pathlib
import os
import pandas as pd
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import umap
import hdbscan
args = parse_train_args()
args.device = 'cuda'if torch.cuda.is_available() else 'cpu'
args.train_eval = 'eval'
if __name__ == '__main__':
    dataset_dir = os.path.join(os.getcwd(),'datasets',args.dataset)
    test_data = pd.read_csv(dataset_dir+f'/process_train_{args.dataset_size}.csv', index_col=False)
    documents = test_data['text'].tolist()
    model = Top2Vec(documents, embedding_model='universal-sentence-encoder')
    umap_args = {'n_neighbors': 100,
                 'min_dist': 0.1,
                 'n_components': 2,
                 'metric': 'cosine'}
    hdbscan_args = {'min_cluster_size': 15,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'}
    umap_model = umap.UMAP(**umap_args).fit(model.document_vectors)
    # X_embedded = TSNE(n_components=2, learning_rate='auto',
    #                  init='random', perplexity=3).fit_transform(model.document_vectors)
    cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)
    color = ['b','g','r','c','m','y','#d36500','#3c728e','#3ce18e','#ffa68e','#6aa68e','#0d86fc','#700d06','#709e06','k']
    for i in range(len(umap_model.embedding_)):
        if cluster.labels_[i] == -1:
            continue
        clr = color[cluster.labels_[i]]
        x,y = umap_model.embedding_[i]
        plt.plot(x,y,color=clr,marker='o')
    plt.savefig(dataset_dir+f'/image_train_{args.dataset_size}.png')
    plt.show()