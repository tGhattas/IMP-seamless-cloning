
from utils import read_image, blending_example1
from scipy.ndimage.filters import convolve
from scipy.sparse import coo_matrix, dok_matrix, lil_matrix, block_diag
from scipy.sparse.linalg import spsolve
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2

# useful kernels
four_neighbors_kernel = [[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]]
four_neighbors_kernel_with_center = [[0, 1, 0],
                                     [1, 1, 1],
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
    dialates img and take diff between original img and dilated img
    :param img: equals the mask float np 2d array
    :return:
    """
    dilated = ndimage.binary_dilation(img, structure=four_neighbors_kernel_with_center).astype(np.float64)
    return dilated - img


def get_basic_vector_field(img, mask):
    """
    calculate sum of v_pq in 4-connected components as defined in term (11) of the paper
    :param img: float np 2d array
    :param mask: float np 2d array
    :return: 2d array where in every entry it has the summation result
    """
    tmp = convolve(img, basic_vector_field_kernel, mode='constant', cval=0.0)
    return tmp #* mask


def _to_flat_coo(r, c, cols_num):
    return r * cols_num + c


def mark_neighbour_variables(pixel_coo, pixel_var_ind, reference, sparse_matrix):
    """

    :param pixel_coo:
    :param pixel_var_ind:
    :param reference:
    :param sparse_matrix:
    :return:
    """
    r, c = pixel_coo
    if r-1 > 0 and reference[r-1, c]:
        sparse_matrix[pixel_var_ind, _to_flat_coo(r-1, c, reference.shape[1])] = -1.0
    if r+1 < reference.shape[0] and reference[r+1, c]:
        sparse_matrix[pixel_var_ind, _to_flat_coo(r+1, c, reference.shape[1])] = -1.0
    if c-1 > 0 and reference[r, c-1]:
        sparse_matrix[pixel_var_ind, _to_flat_coo(r, c-1, reference.shape[1])] = -1.0
    if c+1 < reference.shape[1] and reference[r, c+1]:
        sparse_matrix[pixel_var_ind, _to_flat_coo(r, c+1, reference.shape[1])] = -1.0


def apply_offset(offset, source, target, mask):
    """

    :param offset:
    :param source:
    :param target:
    :param mask:
    :return:
    """
    y_max, x_max = target.shape
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min

    M = np.float64([[1, 0, offset[0]], [0, 1, offset[1]]])
    warped_source = cv2.warpAffine(source, M, (x_range, y_range))

    mask = mask[y_min:y_max, x_min:x_max]
    return warped_source, mask, y_max, x_max, y_min, x_min, x_range, y_range

def get_laplacian_mat(n, m):

    """
    taken from Git *** https://github.com/PPPW/poisson-image-editing
    Generate the Poisson matrix.

    Refer to:
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix
    """
    mat_D = lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)

    return mat_A


def seamless_cloning_single_channel(source, target, mask, offset, gradient_field_source_only=True):
    """
    :param source:
    :param target:
    :param mask:
    :param offset:
    :param gradient_field_source_only:
    :return:
    """
    source, mask, y_max, x_max, y_min, x_min, x_range, y_range = apply_offset(offset, source, target, mask)
    Np = get_4_neigbours_amount(target) * mask  # for the calc of left first term in equation (7)
    omega_boundary = get_omega_boundary(mask)

    # sum_f_star = convolve(omega_boundary * target, four_neighbors_kernel, mode='constant', cval=0.0) * mask  # for the
    # # calc of right first term in equation (7)
    # vector_field_sum = get_basic_vector_field(source, mask)  # for the calc of right second term in equation (7)
    # eq_right = sum_f_star + vector_field_sum

    laplacian = get_laplacian_mat(y_range, x_range)
    g = source.flatten()
    eq_right = laplacian.dot(g)

    flat_mask = mask.flatten()
    flat_mask_ind = np.where(flat_mask > 0)  # returns all pixels indices

    flat_eq_left_1 = Np.flatten()
    diag_indices = (np.arange(flat_mask_ind[0].shape[-1]), np.arange(flat_mask_ind[0].shape[-1]))
    eq_left_1_sys = coo_matrix((flat_eq_left_1[flat_mask_ind], diag_indices))
    eq_left_2_reference = ndimage.binary_dilation(Np, structure=four_neighbors_kernel).astype(np.float64) * mask
    eq_left_sys = dok_matrix(eq_left_1_sys)

    # go over pixels and check
    for pixel_var_ind in diag_indices[-1]:
        pixel_coo = pixel_var_ind // Np.shape[1], pixel_var_ind % Np.shape[1]
        mark_neighbour_variables(pixel_coo, pixel_var_ind, eq_left_2_reference, eq_left_sys)

    eq_left_sys = eq_left_sys.tocsr()

    flat_eq_right = eq_right.flatten()
    print(eq_left_sys.shape)
    print(flat_eq_right[flat_mask_ind].shape)
    print('max', np.max(flat_eq_right))
    f = spsolve(eq_left_sys, flat_eq_right[flat_mask_ind]).astype(np.float64)
    print(f)

    # reconstruct image
    blend = target.flatten()

    blend[flat_mask_ind] = f + g[flat_mask_ind]
    blend = blend.reshape(target.shape)
    # blend[omega_boundary>0] = target[omega_boundary>0]-source[omega_boundary>0]
    return np.int_(blend.clip(0, 1) * 255).astype('uint8')


def seamless_cloning(source, target, mask, offset=(0, 0), gradient_field_source_only=True):
    """

    :param source:
    :param target:
    :param mask:
    :param offset:
    :param gradient_field_source_only:
    :return:
    """


    mask = mask > 0.1
    R = seamless_cloning_single_channel(source[...,0], target[...,0], mask, offset, gradient_field_source_only)
    G = seamless_cloning_single_channel(source[...,1], target[...,1], mask, offset, gradient_field_source_only)
    B = seamless_cloning_single_channel(source[...,2], target[...,2], mask, offset, gradient_field_source_only)
    result = np.zeros_like(target, dtype='uint8')
    result[:,:, 0] = R
    result[:,:, 1] = G
    result[:,:, 2] = B
    return result

if __name__ == '__main__':
    # target = read_image('./external/main-1.jpg', 2)
    # source = read_image('./external/blend-1.jpg', 2)
    # mask = read_image('./external/mask-1.jpg', 1)

    target = read_image('./external/target1.jpg', 2)
    source = read_image('./external/source1.jpg', 2)
    mask = read_image('./external/mask1.png', 1)


    # target_g = read_image('./external/main-1.jpg', 1)
    # source_g = read_image('./external/blend-1.jpg', 1)
    # cloned_single = seamless_cloning_single_channel(source_g, target_g, mask)
    # plt.imshow(cloned_single), plt.show()
    offset = (0, 66)
    cloned = seamless_cloning(source, target, mask, offset=offset)
    plt.imshow(cloned), plt.show()
    # basic = target.copy()
    # basic[mask>0.1] = source[mask>0.1]
    # plt.imshow(basic), plt.show()#, cmap=plt.cm.gray
    # blending_example1()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
