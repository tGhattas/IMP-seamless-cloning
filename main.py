from utils import read_image, pyramid_blending_example1, plot
from scipy.ndimage.filters import convolve
from scipy.sparse import lil_matrix, block_diag
from scipy.sparse.linalg import spsolve
from scipy.ndimage.morphology import distance_transform_edt
from scipy import ndimage, fft, signal
from tqdm import tqdm
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


def get_basic_vector_field(img):
    """
    calculate sum of v_pq in 4-connected components as defined in term (11) of the paper
    :param img: float np 2d array
    :param mask: float np 2d array
    :return: 2d array where in every entry it has the summation result
    """
    tmp = convolve(img, basic_vector_field_kernel, mode='constant', cval=0.0)
    return tmp


def make_identity_off_mask(mask, mat, y_range, x_range):
    """
    :param mask: binary mask defining f function
    :param mat: sparse matrix of the left hand side equation system
    :param y_range: obtained from apply_offset
    :param x_range: obtained from apply_offset
    :return:
    """
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                ind = x + y * x_range
                mat[ind, ind] = 1
                mat[ind, ind + 1] = 0
                mat[ind, ind - 1] = 0
                mat[ind, ind + x_range] = 0
                mat[ind, ind - x_range] = 0


def apply_offset(offset, source, target, mask):
    """
    Warp source according to offset.
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

    laplacian = get_laplacian_mat(y_range, x_range)
    flat_source = source[y_min:y_max, x_min:x_max].flatten()
    flat_target = target[y_min:y_max, x_min:x_max].flatten()
    flat_mask = mask.flatten()

    eq_left_sys = laplacian.tocsc()

    # inside f
    eq_right = laplacian.dot(flat_source)

    flat_eq_right = eq_right.flatten()

    # outside f
    flat_eq_right[flat_mask == 0] = flat_target[flat_mask == 0]
    make_identity_off_mask(mask, eq_left_sys, y_range, x_range)

    s = spsolve(eq_left_sys, flat_eq_right).astype(np.float64)

    # reconstruct image
    blend = s.reshape(target.shape)
    blend = (blend.clip(0, 1) * 255).astype('uint8')

    return blend


def seamless_cloning(source, target, mask, offset=(0, 0), gradient_field_source_only=True):
    """
    Based on Poisson solver
    :param source:
    :param target:
    :param mask:
    :param offset:
    :param gradient_field_source_only:
    :return:
    """
    mask = mask > 0.1
    mask = mask.astype('uint8')
    result = np.zeros_like(target, dtype='uint8')
    for channel in tqdm(range(len('RGB')), desc='Possion seamless cloning RGB'):
        result[..., channel] = seamless_cloning_single_channel(source[..., channel], target[..., channel], mask, offset,
                                                                   gradient_field_source_only)

    return result


def create_mask_dist_transform(mask):
    """
    Creates a Shepard's interpolation kernel
    :param mask: binary mask
    :return: kernel
    """
    sm = mask.shape
    kernel = np.ones_like(mask, dtype='float64')
    kernel[round(sm[0] / 2), round(sm[1] / 2)] = 0.0
    kernel = distance_transform_edt(kernel)
    kernel = 1 / ((kernel + 0.1) ** 3)
    return kernel

def apply_filter_fft(image, kernel):
    ft_image = fft.fft2(image)
    ft_image = signal.convolve2d(ft_image, kernel, mode='same')
    return fft.ifft2(ft_image).real


def shepards_single_channel(source, target, mask, offset, F):
    """
    code for Convolution Pyramid. link: https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/
    :param source:
    :param target:
    :param mask:
    :param offset:
    :param F:
    :return:
    """
    source, mask, y_max, x_max, y_min, x_min, x_range, y_range = apply_offset(offset, source, target, mask)

    difference = target - source
    boundary = get_omega_boundary(mask)
    difference[boundary == 0] = 0
    # Shepard Interpolation Convolution
    filtered_diff = apply_filter_fft(difference, F)
    filtered_boundary = apply_filter_fft(boundary, F)
    temp = filtered_diff / filtered_boundary + source

    blend = target.copy()
    mask = mask > 0
    blend[mask] = temp[mask]
    blend = (blend.clip(0, 1) * 255).astype('uint8')
    return blend


def shepards_seamless_cloning(source, target, mask, offset=(0, 0), F):
    """
    Based on Poisson solver
    :param source:
    :param target:
    :param mask:
    :param offset:
    :param gradient_field_source_only:
    :return:
    """
    mask = mask > 0.1
    mask = mask.astype('uint8')
    result = np.zeros_like(target, dtype='uint8')
    F = create_mask_dist_transform(mask)
    for channel in tqdm(range(len('RGB')), desc="Shepard's seamless cloning RGB"):
        result[..., channel] = shepards_single_channel(source[..., channel], target[..., channel], mask, offset, F)

    return result


def shepards_blending_example1():
    target = read_image('./external/main-1.jpg', 2)
    source = read_image('./external/blend-1.jpg', 2)
    mask = read_image('./external/mask-1.jpg', 1)
    offset = (0, 0)
    cloned = shepards_seamless_cloning(source, target, mask, offset, None)
    plt.imshow(cloned), plt.show()
    plot(source, target, mask, cloned, title="Shepard's Based Blending 1")


def poisson_blending_example1(monochromatic_source=True):
    target = read_image('./external/main-1.jpg', 2)
    # make source monochromatic to avoid color darkening in results
    if monochromatic_source:
        source = read_image('./external/blend-1.jpg', 1)
        mono_source = np.zeros(source.shape+(3,), dtype=np.float64)
        for _ in range(len('RGB')):
            mono_source[..., _] = source
        source = mono_source
    else:
        source = read_image('./external/blend-1.jpg', 2)
    mask = read_image('./external/mask-1.jpg', 1)
    offset = (0, 0)
    cloned = seamless_cloning(source, target, mask, offset=offset)
    plt.imshow(cloned), plt.show()
    plot(source, target, mask, cloned, title='Possion Based Blending 1')


def poisson_blending_example2():
    target = read_image('./external/target1.jpg', 2)
    source = read_image('./external/source1.jpg', 2)
    mask = read_image('./external/mask1.png', 1)
    offset = (0, 66)
    cloned = seamless_cloning(source, target, mask, offset=offset)
    plt.imshow(cloned), plt.show()
    plot(source, target, mask, cloned, title='Possion Based Blending 2')


if __name__ == '__main__':
    shepards_blending_example1()
    # poisson_blending_example1()
    # poisson_blending_example1(monochromatic_source=False)
    # pyramid_blending_example1()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
