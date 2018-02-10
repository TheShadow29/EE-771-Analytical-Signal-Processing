import numpy as np


def find_num_zero_cross(vec):
    num_zero_cross = 0
    for i in range(vec.shape[0]-1):
        if vec[i] * vec[i+1] < 0:
            num_zero_cross += 1
    return num_zero_cross


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

    # q1a
    print('Eig Vec Basis', eig_vec)
    # q1b
    f_sign = np.array([3, 1, 2, 2, 2, 1, 1], np.float_)

    f_sign_new_basis = np.dot(eig_vec.T, f_sign)
    print('New Coordinate Basis', f_sign_new_basis)

    # q1c
    num_zero_cross_vec = np.zeros(eig_vec.shape[1])
    for i in range(eig_vec.shape[1]):
        num_zero_cross_vec[i] = find_num_zero_cross(eig_vec[:, i])
    print('Zero Crossing Vector', num_zero_cross_vec)

    # q1d
    f_sign_new_basis_lp = np.dot(np.diag([1, 1, 1, 1, 0, 0, 0]), f_sign_new_basis)
    f_recon_lp = np.dot(eig_vec, f_sign_new_basis_lp)
    print('Graph signal', f_sign)
    print('Low Pass graph signal', f_recon_lp)
