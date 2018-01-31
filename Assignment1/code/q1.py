import numpy as np

if __name__ == "__main__":
    W_mat = np.zeros((7, 7))
    W_mat[0, 1] = W_mat[0, 2] = W_mat[2, 3] = W_mat[3, 4] = W_mat[3, 5] = W_mat[3, 6] = 1
    W_mat = W_mat + W_mat.T

    D_mat = np.diag(np.sum(W_mat, axis=1))

    L_mat = D_mat - W_mat
    eig_val, eig_vec = np.linalg.eig(L_mat)
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    print(eig_vec)
