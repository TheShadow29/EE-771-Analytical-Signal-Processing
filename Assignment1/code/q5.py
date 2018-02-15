import numpy as np


def norm_laplacian(W_mat):
    # W_mat = nx.adjacency_matrix(g1).todense()
    D_mat = np.diag(np.ravel(np.sum(W_mat, axis=1)))
    # D_root = np.diag(np.sum(W_mat, axis=1))
    D_root = np.sqrt(D_mat)
    D_root_inv = np.linalg.inv(D_root)
    # L_mat = D_mat - W_mat
    # pdb.set_trace()
    L_mat = np.eye(W_mat.shape[0]) - np.dot(D_root_inv, np.dot(W_mat, D_root_inv))
    eig_val, eig_vec = np.linalg.eig(L_mat)
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    return L_mat, eig_vec, eig_val


if __name__ == "__main__":
    W_mat = np.zeros((6, 6))
    W_mat[0, 3] = W_mat[0, 4] = 1
    W_mat[1, 4] = W_mat[1, 5] = 1
    W_mat[2, 3] = W_mat[2, 5] = 1
    W_mat[3, 0] = W_mat[3, 2] = 1
    W_mat[4, 0] = W_mat[4, 1] = 1
    W_mat[5, 1] = W_mat[5, 2] = 1

    f_signal = np.array([1, 2, 3, 1, 2, 3])

    H = [1, 2, 3]
    L = [4, 5, 6]

    L_mat, eig_vec, eig_val = norm_laplacian(W_mat)
    # verified

    beta_H = np.array([1, 1, 1, -1, -1, -1])
    beta_L = -beta_H

    J_bL = np.diag(beta_L)
    J_bH = np.diag(beta_H)

    fL_ds = 0.5 * np.dot((np.eye(J_bL.shape[0]) + J_bL), np.array(f_signal))
    fH_ds = 0.5 * np.dot((np.eye(J_bH.shape[0]) + J_bH), np.array(f_signal))
