import matplotlib.pyplot as plt
import numpy as np
import scipy.special as scs


def get_stability_const(K):
    L = 10
    cmin = 0.1
    delta1 = 0.1
    twoK1 = 2*K + 1
    twoK = 2*K
    halfK = K // 2
    rootK = np.sqrt(K)
    eps_y = 1
    eps_x = (twoK1 / twoK)**K / np.sqrt(twoK1) * eps_y
    norm_a = np.sqrt(scs.comb(2*K, K))
    inv_x = 1
    for v in range(1, halfK + 1):
        inv_x *= 4 * np.sin(np.pi * v * delta1)**2
    tmp_const = inv_x
    inv_x = inv_x / (rootK * 2**(K-1))
    inv_x = cmin * inv_x ** 2
    inv_x = 1 / inv_x
    eps_a = 2 * (1 + rootK * norm_a) * inv_x * eps_x
    stab_const = (rootK * K * eps_a) / (tmp_const)
    stab_const_fin = 2**(-L) * stab_const
    print(stab_const_fin)
    return stab_const_fin


if __name__ == "__main__":
    stab_const_list = []
    for k in range(1, 6):
        stab_const_tmp = get_stability_const(k)
        stab_const_list.append(stab_const_tmp)
    plt.plot(np.arange(1, 6), stab_const_list)
    # plt.show()
    plt.savefig('plotek.png')
