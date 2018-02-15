from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pdb


def er_graph(n, p):
    # n is number of nodes
    # p is probability
    return nx.erdos_renyi_graph(n, p)


def norm_laplacian(g):
    W_mat = nx.adjacency_matrix(g1).todense()
    D_mat = np.diag(np.ravel(np.sum(W_mat, axis=1)))
    # D_root = np.diag(np.sum(W_mat, axis=1))
    D_root = np.sqrt(D_mat)
    D_root_inv = np.linalg.inv(D_root)
    # L_mat = D_mat - W_mat
    pdb.set_trace()
    L_mat = np.eye(W_mat.shape[0]) - np.dot(D_root_inv, np.dot(W_mat, D_root_inv))
    eig_val, eig_vec = np.linalg.eig(L_mat)
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    return eig_val


if __name__ == "__main__":
    n = 100
    p1 = 0.1
    p2 = 0.2
    g1 = er_graph(n, p1)
    g2 = er_graph(n, p2)
    eig_val1 = norm_laplacian(g1)
    eig_val2 = norm_laplacian(g2)

    plt.plot(eig_val1)
    plt.plot(eig_val2)
    plt.show()
