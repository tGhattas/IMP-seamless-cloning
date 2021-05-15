import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray


def read_image(filename, representation):
    '''
    :param filename: file path to load
    :param representation: 1 - grey , 2 - RGB
    :return: retuns ndarray of the image
    '''
    if representation not in (1, 2):
        raise Exception("representation code should be either 1 or 2 defining whether the output should be a grayscale"
                        " image (1) or an RGB image (2)")
    im = imageio.imread(filename)
    im = im.astype(np.float64) / 255
    if representation == 1:
        return rgb2gray(im)
    return im


def _get_gaussian_filter(filter_size):
    filter_main = np.array([1, 1], dtype=np.float64)
    filter_ = filter_main.copy()
    for i in range(filter_size - 2):
        filter_ = np.convolve(filter_, filter_main)
    filter_vec = filter_ / max(1.0, float(np.sum(filter_)))
    return filter_vec.reshape(1, filter_vec.shape[0])


def _reduce_image(im, filter_vec):
    blured = convolve(im, filter_vec)
    blured = convolve(blured, filter_vec.T)
    reduced = blured[::2, ::2]
    return reduced, blured


def _expand_image(im, filter_vec, g_pyr_lvl=None):
    expanded = np.zeros((2*im.shape[0], 2*im.shape[1]))
    expanded[::2, ::2] = im
    blured = convolve(expanded, 2 * filter_vec)
    blured = convolve(blured, 2 * filter_vec.T)
    if g_pyr_lvl is not None:
        padded = np.pad(g_pyr_lvl, [(0, max(0, blured.shape[0] - g_pyr_lvl.shape[0])),
                                    (0, max(0, blured.shape[1] - g_pyr_lvl.shape[1]))],
                        mode='constant')
        laplacian = padded - blured
        return laplacian
    else:
        return blured


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the representation set to 1).
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
    :return:
    """
    filter_vec = _get_gaussian_filter(filter_size)
    pyr = [im.copy()]
    for i in range(max_levels-1):
        reduced, blured = _reduce_image(pyr[-1], filter_vec)
        if reduced.shape[0] >= 16 and reduced.shape[1] >= 16:
            pyr.append(reduced)
        else:
            break
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    pyr = []
    g_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(g_pyr)-1):
        im = g_pyr[i+1]
        laplacian = _expand_image(im, filter_vec, g_pyr[i])
        pyr.append(laplacian)
    pyr.append(g_pyr[-1])
    return pyr, filter_vec


def render_pyramid(pyr, levels):
    pyt_padded = []
    first_height = pyr[0].shape[0]
    for i in range(min(len(pyr), levels)):
        max_val = np.max(pyr[i])
        stretched = pyr[i] / max_val if max_val else pyr[i]
        padded = np.pad(stretched, [(0, first_height-pyr[i].shape[0]), (0, 0)], mode='constant')
        pyt_padded.append(padded)
    res = np.hstack(tuple(pyt_padded))
    return res


def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap=cm.gray)
    plt.show()


def laplacian_to_image(lpyr, filter_vec, coeff):
    res = coeff[-1] * lpyr[-1]
    for i in range(1, len(lpyr)):
        expanded = coeff[-i] * _expand_image(res, filter_vec)
        small, big = (expanded, lpyr[-i-1]) if expanded.shape[0] < lpyr[-i-1].shape[0] else (lpyr[-i-1], expanded)
        small = np.pad(small, [(0, abs(lpyr[-i-1].shape[0] - expanded.shape[0])),
                               (0, abs(lpyr[-i-1].shape[1] - expanded.shape[1]))],
                       mode='constant')
        res = small + big
    return res


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    l_1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l_2, _ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_f = mask.astype(np.float64)
    g_m, _ = build_gaussian_pyramid(mask_f, max_levels, filter_size_mask)
    l_out = []
    for i in range(len(l_1)):
        lvl_blend = l_1[i]*g_m[i] + (1-g_m[i])*l_2[i]
        l_out.append(lvl_blend)
    im_blend = laplacian_to_image(l_out, filter_vec, [1]*len(l_out))
    return im_blend.clip(0.0, 1.0)


def _blend_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    blend = np.zeros_like(im1)
    for i in range(len('RGB')):
        blend[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels, filter_size_im,
                                          filter_size_mask)
    return blend


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def plot(im1, im2, mask, im_blend, title='Pyramid Blending'):
    plt.figure(title)

    plt.subplot(141)
    plt.imshow(im1)
    plt.title('Image 1')

    plt.subplot(142)
    plt.imshow(im2)
    plt.title('Image 2')

    plt.subplot(143)
    plt.imshow(mask, cmap=cm.gray)
    plt.title('Mask')

    plt.subplot(144)
    plt.imshow(im_blend)
    plt.title('Blend')

    plt.show()

    plt.imshow(im_blend)
    plt.show()


def pyramid_blending_example1():
    im1 = read_image(relpath('external/blend-1.jpg'), 2)
    im2 = read_image(relpath('external/main-1.jpg'), 2)
    mask = read_image(relpath('external/mask-1.jpg'), 1)
    mask = np.round(mask).astype(np.bool)
    im_blend = _blend_rgb(im1, im2, mask, 13, 3, 5)
    plot(im1, im2, mask, im_blend, 'Hired in NASA')
    return im1, im2, mask, im_blend


def pyramid_blending_example2():
    im1 = read_image(relpath('external/blend-2.jpg'), 2)
    im2 = read_image(relpath('external/main-2.jpg'), 2)
    mask = read_image(relpath('external/mask-2.jpg'), 1)
    mask = np.round(mask).astype(np.bool)
    im_blend = _blend_rgb(im1, im2, mask, 13, 3, 3)
    plot(im1, im2, mask, im_blend, 'Lava Under The Bridge - Haifa')
    return im1, im2, mask, im_blend