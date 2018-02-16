from __future__ import division
import numpy as np
from pathlib import Path
import skimage.io as skio
# import networkx as nx
# import skimage.transform as skt
import skimage.color as skc
import skimage.measure as skm
# import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
# import pdb


def show_images(images, cols=1, titles=None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def dist(i, j, p, theta, k):
    r1, c1 = i // p, i % p
    r2, c2 = j // p, j % p
    # d = np.exp(-((r1 - r2)**2 + (c1 - c2)**2) / 2 * theta**2)
    d = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
    return d


def get_laplacian(p, theta=1.3, k=1.5):
    W_mat = np.zeros((p*p, p*p), dtype=np.float_)
    for i in range(W_mat.shape[0]):
        for j in range(W_mat.shape[1]):
            dij = dist(i, j, p, theta, k)
            assert np.isreal(dij)
            if i != j and dij <= k:
                W_mat[i, j] = np.exp(-dij**2/(2 * theta**2))

    D_mat = np.diag(np.ravel(np.sum(W_mat, axis=1)))
    L_mat = D_mat - W_mat
    # pdb.set_trace()
    assert np.allclose(L_mat, L_mat.T)
    return L_mat


def denoiser(orig_img_path, noisy_img_path):
    start_time = time.time()
    orig_img = skc.rgb2gray(skio.imread(orig_img_path))
    noisy_img = skc.rgb2gray(skio.imread(noisy_img_path))

    p = orig_img.shape[0] // 32  # =16

    # orig_img_ds = skt.downscale_local_mean(orig_img, (8, 8))
    # noisy_img_ds = skt.downscale_local_mean(noisy_img, (8, 8))

    # grid_graph = nx.grid_2d_graph(p, p)
    # lap = nx.laplacian_matrix(grid_graph)
    # lap = lap.astype(np.float_)
    # lap = lap.todense()
    lap = get_laplacian(p)
    # eig_val, eig_vec = sparse.linalg.eigs(lap, k=p*p)
    eig_val, eig_vec = np.linalg.eig(lap)
    # eig_val = np.real(eig_val)
    # eig_vec = np.real(eig_vec)
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    gamma = 2
    div_vec = 1 + gamma * eig_val

    # noisy_img_ds_1d = np.ravel(noisy_img_ds)
    out_img = np.zeros(orig_img.shape)

    for r in range(0, orig_img.shape[0], p):
        for c in range(0, orig_img.shape[1], p):
            noisy_img_patch_1d = np.ravel(noisy_img[r:r+p, c:c+p])
            y_hat_vec = np.dot(eig_vec.T, noisy_img_patch_1d)
            opt_f_vec = np.divide(y_hat_vec, div_vec)
            opt_f_vec = np.ravel(opt_f_vec)
            denoise_img_ = np.dot(eig_vec, opt_f_vec)
            denoise_img = denoise_img_.reshape(p, p)
            out_img[r:r+p, c:c+p] = denoise_img
            # pdb.set_trace()

    end_time = time.time()

    print("--- %s seconds ---" % (end_time - start_time))

    # psnr_before = s
    # skio.imshow(out_img, cmap='gray')
    # skio.show()
    out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())

    psnr_before = skm.compare_psnr(orig_img, noisy_img)
    psnr_after = skm.compare_psnr(orig_img, out_img)
    print('Before ', psnr_before, 'After', psnr_after)

    # skio.imshow_collection([orig_img, noisy_img, out_img])
    # skio.show()

    # pdb.set_trace()
    return orig_img, noisy_img, out_img, psnr_before, psnr_after


if __name__ == "__main__":

    img_dir = Path('../data/denoise')
    orig_img_path1 = img_dir / 'gull.jpg'
    noisy_img_path1 = img_dir / 'gull-noise.jpg'

    orig_img_path2 = img_dir / 'sfo.jpg'
    noisy_img_path2 = img_dir / 'sfo-noise.jpg'

    orig_img1, noisy_img1, out_img1, psnr_before1, psnr_after1 = denoiser(orig_img_path1,
                                                                          noisy_img_path1)
    titles1 = ['original image', 'noisy img psnr=' + str(psnr_before1),
               'denoised image psnr=' + str(psnr_after1)]
    orig_img2, noisy_img2, out_img2, psnr_before2, psnr_after2 = denoiser(orig_img_path2,
                                                                          noisy_img_path2)
    titles2 = ['original image', 'noisy img psnr=' + str(psnr_before2),
               'denoised image psnr=' + str(psnr_after2)]

    show_images([orig_img1, noisy_img1, out_img1], titles=titles1)
    show_images([orig_img2, noisy_img2, out_img2], titles=titles2)
