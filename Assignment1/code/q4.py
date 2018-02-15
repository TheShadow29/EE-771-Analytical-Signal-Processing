from __future__ import division
import numpy as np
from pathlib import Path
import skimage.io as skio
# import networkx as nx
import skimage.transform as skt
import skimage.color as skc
# import scipy.sparse as sparse
import time
import pdb


def dist(i, j, p, theta, k):
    r1, c1 = i // p, i % p
    r2, c2 = j // p, j % p
    # d = np.exp(-((r1 - r2)**2 + (c1 - c2)**2) / 2 * theta**2)
    d = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
    return d


def get_laplacian(p, theta=0.1, k=1):
    W_mat = np.zeros((p*p, p*p), dtype=np.float_)
    for i in range(W_mat.shape[0]):
        for j in range(W_mat.shape[1]):
            dij = dist(i, j, p, theta, k)
            if i != j and dij <= k:
                W_mat[i, j] = np.exp(-dij/(2 * theta**2))

    D_mat = np.diag(np.ravel(np.sum(W_mat, axis=1)))
    L_mat = D_mat - W_mat
    # pdb.set_trace()
    return L_mat


def denoiser(orig_img_path, noisy_img_path):
    start_time = time.time()
    orig_img = skc.rgb2gray(skio.imread(orig_img_path))
    noisy_img = skc.rgb2gray(skio.imread(noisy_img_path))

    p = orig_img.shape[0] // 8

    orig_img_ds = skt.downscale_local_mean(orig_img, (8, 8))
    noisy_img_ds = skt.downscale_local_mean(noisy_img, (8, 8))

    # grid_graph = nx.grid_2d_graph(p, p)
    # lap = nx.laplacian_matrix(grid_graph)
    # lap = lap.astype(np.float_)
    # lap = lap.todense()
    lap = get_laplacian(p)
    # eig_val, eig_vec = sparse.linalg.eigs(lap, k=p*p)
    eig_val, eig_vec = np.linalg.eig(lap)
    eig_val = np.real(eig_val)
    eig_vec = np.real(eig_vec)
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    gamma = 10

    div_vec = 1 + gamma * eig_val
    noisy_img_ds_1d = np.ravel(noisy_img_ds)

    y_hat_vec = np.dot(eig_vec.T, noisy_img_ds_1d)

    opt_f_vec = np.divide(y_hat_vec, div_vec)
    opt_f_vec = np.ravel(opt_f_vec)
    denoise_img_ = np.dot(eig_vec, opt_f_vec)
    denoise_img = denoise_img_.reshape(p, p)

    end_time = time.time()

    print("--- %s seconds ---" % (end_time - start_time))
    skio.imshow(denoise_img)
    skio.show()


if __name__ == "__main__":

    img_dir = Path('../data/denoise')
    orig_img_path1 = img_dir / 'gull.jpg'
    noisy_img_path1 = img_dir / 'gull-noise.jpg'

    orig_img_path2 = img_dir / 'sfo.jpg'
    noisy_img_path2 = img_dir / 'sfo-noise.jpg'
    # denoiser(orig_img_path1, noisy_img_path1)
    denoiser(orig_img_path2, noisy_img_path2)
