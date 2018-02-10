import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def er_graph(n, p):
    # n is number of nodes
    # p is probability
    return nx.erdos_renyi_graph(n, p)


def find_num_zero_cross(vec):
    num_zero_cross = 0
    for i in range(vec.shape[0]-1):
        if vec[i] * vec[i+1] < 0:
            num_zero_cross += 1
    return num_zero_cross


def zero_cross_plot(g):
    W_mat = nx.adjacency_matrix(g1).todense()
    D_mat = np.diag(np.sum(W_mat, axis=1))

    L_mat = D_mat - W_mat
    eig_val, eig_vec = np.linalg.eig(L_mat)
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    num_zero_cross_vec = np.zeros(eig_vec.shape[1])
    for i in range(eig_vec.shape[1]):
        num_zero_cross_vec[i] = find_num_zero_cross(eig_vec[:, i])
    # print('Zero Crossing Vector', num_zero_cross_vec)
    return num_zero_cross_vec


if __name__ == "__main__":
    n = 100
    p1 = 0.1
    p2 = 0.2
    g1 = er_graph(n, p1)
    g2 = er_graph(n, p2)
    zero_vec1 = zero_cross_plot(g1)
    zero_vec2 = zero_cross_plot(g2)
    plt.plot(zero_vec1, '-r')
    plt.plot(zero_vec2, '-b')
    plt.show()
    # Conclusion : Same number of zero cross
