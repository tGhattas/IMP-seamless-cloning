
from utils import read_image
from scipy.ndimage.filters import convolve
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np

# useful kernels
four_neighbors_kernel = [[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]]
basic_vector_field_kernel = [[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]]


def get_4_neigbours_amount(img):
    """
    convolves img in order to calculate the 4-connected neighbors which are in S
    :param img: float np 2d array
    :return: 2d np array equals |N_p| in the pixel (2d-coords) p
    """
    ones = np.ones_like(img)
    return convolve(ones, four_neighbors_kernel, mode='constant', cval=0.0)


def get_omega_boundary(img):
    """
    convolves img in order to calculate the discrete omega boundary as defined in the paper
    :param img: equals the mask float np 2d array
    :return: 2d np array equals |N_p| in the pixel (2d-coords) p
    """
    s_without_omega = 1.0-img  # assuming img is a binary mask
    return np.float_(4 > convolve(s_without_omega, four_neighbors_kernel, mode='nearest') > 0)


def get_basic_vector_field(img, mask):
    """
    calculate sum of v_pq in 4-connected components as defined in term (11) of the paper
    :param img: float np 2d array
    :param mask: float np 2d array
    :return: 2d array where in every entry it has the summation result
    """
    tmp = convolve(img, basic_vector_field_kernel, mode='constant', cval=0.0)
    return tmp * mask


def seamless_cloning(source, target, mask, offset=None, gradient_field_source_only=True):
    """
    :param source:
    :param target:
    :param mask:
    :param offset:
    :param gradient_field_source_only:
    :return:
    """
    Np = get_4_neigbours_amount(target) * mask  # for the calc of left first term in equation (7)

    omega_boundary = get_omega_boundary(mask)

    sum_f_star = convolve(omega_boundary * target, four_neighbors_kernel, mode='constant', cval=0.0) * mask  # for the
    # calc of right first term in equation (7)

    vector_field_sum = get_basic_vector_field(source, mask)  # for the calc of right second term in equation (7)

    eq_right = sum_f_star + vector_field_sum

    Np = Np * mask
    flat_Np = Np.flatten()
    flat_mask = mask.flatten()
    flat_mask_ind = np.where(flat_mask)
    inds = (np.arange(len(flat_mask_ind[0].shape[-1])),) + flat_mask_ind
    sm = coo_matrix((flat_Np[np.int_(flat_mask)], inds))


if __name__ == '__main__':
    img = read_image('./external/main-1.jpg', 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
