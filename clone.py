from utils import read_image, pyramid_blending_example1, plot, plt
from scipy.ndimage.filters import convolve
from scipy.sparse import lil_matrix, block_diag
from scipy.sparse.linalg import spsolve
from scipy.ndimage.morphology import distance_transform_edt
from scipy import ndimage, fft
from tqdm import tqdm
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
    y_max, x_max = target.shape[:2]
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


def get_grad_magnitude(img):
    """
    returns the magnitude in float 64
    :param img:
    :return:
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    mag = sobelx**2 + sobely**2
    return mag


def seamless_cloning_single_channel(source, target, mask, offset, gradient_field_source_only):
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

    if gradient_field_source_only:
        # inside f
        eq_right = laplacian.dot(flat_source)
    else:
        # corresponding to vector field from equation (11) in the paper.
        grad_g = get_grad_magnitude(source)
        grad_f_star = get_grad_magnitude(target)
        cond = np.abs(grad_f_star) > np.abs(grad_g)
        eq_right = np.where(cond, target, source)
        eq_right = laplacian.dot(eq_right.flatten())

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


def apply_filter(image, kernel, in_freq_domain=False):
    if in_freq_domain:
        ft_kernel = fft.fft2(kernel)
        ft_image = fft.fft2(image)
        ft_image = ft_image * ft_kernel
        return fft.ifft2(ft_image).real
    else:
        return cv2.filter2D(image, -1, kernel)


def shepards_single_channel(source, target, mask, offset, F):
    """
    Based on code for "Convolution Pyramid" - Farbman et al.
    link: https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/
    :param source:
    :param target:
    :param mask: binary mask
    :param offset: tuple of size 2 representing the offset
    :param F: kernel to use in Shepard's interpolation
    :return:
    """

    difference = target - source
    boundary = get_omega_boundary(mask)
    difference[boundary == 0] = 0
    # Shepard Interpolation Convolution
    filtered_diff = apply_filter(difference, F)
    filtered_boundary = apply_filter(boundary, F)
    temp = filtered_diff / filtered_boundary + source

    blend = target.copy()
    mask = mask > 0
    blend[mask] = temp[mask]
    blend = (blend.clip(0, 1) * 255).astype('uint8')
    blend = blend.reshape(target.shape)
    return blend


def shepards_seamless_cloning(source, target, mask, offset, F=None):
    """
    Based on Poisson solver
    :param source:
    :param target:
    :param mask: binary mask
    :param offset: tuple of size 2 representing the offset
    :param F: kernel to use in Shepard's interpolation
    :return:
    """
    mask = mask > 0.1
    mask = mask.astype('uint8')
    if F is None:
        F = create_mask_dist_transform(mask)
    source, mask, y_max, x_max, y_min, x_min, x_range, y_range = apply_offset(offset, source, target, mask)
    source = source[y_min:y_max, x_min:x_max]
    target = target[y_min:y_max, x_min:x_max]
    mask = mask[y_min:y_max, x_min:x_max]
    result = np.zeros_like(target, dtype='uint8')
    for channel in tqdm(range(len('RGB')), desc="Shepard's seamless cloning RGB"):
        tmp = shepards_single_channel(source[..., channel], target[..., channel], mask, offset, F)
        result[..., channel] = tmp.reshape(mask.shape)

    return result


'''
Note on running times:
    Shepard's based convolution uses cv2.Filter2d which uses the frequency domain to apply the filter, therefore the 
    time complexity of the blending is bound by O(NlogN) where N is the number of pixels.
    However, in Possion based solver, it builds the blend by solving a sparse linear equation using multifrontal LU
    factorization.
    
     On Macbook Pro mid 14, intel i5:
    Shepard's seamless cloning RGB: 100%|██████████| 3/3 [00:00<00:00,  8.56it/s]
    Poisson seamless cloning RGB: 100%|██████████| 3/3 [03:36<00:00, 72.16s/it]
'''

