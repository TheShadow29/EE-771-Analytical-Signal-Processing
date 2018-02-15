from __future__ import division
import numpy as np
from pathlib import Path
import skimage.io as skio
import networkx as nx
import scipy.sparse as sparse
import time


if __name__ == "__main__":
    start_time = time.time()
    img_dir = Path('../data/denoise')
    orig_img_path = img_dir / 'gull.jpg'
    noisy_img_path = img_dir / 'gull-noise.jpg'

    orig_img = skio.imread(orig_img_path)
    noisy_img = skio.imread(noisy_img_path)

    p = orig_img.shape[0] // 8
    grid_graph = nx.grid_2d_graph(p, p)
    lap = nx.laplacian_matrix(grid_graph)
    lap = lap.astype(np.float_)

    # eig_val, eig_vec = sparse.linalg.eigs(lap, k=p*p)
    eig_val, eig_vec = np.linalg.eig(lap.todense())
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    end_time = time.time()

    print("--- %s seconds ---" % (end_time - start_time))

    #
