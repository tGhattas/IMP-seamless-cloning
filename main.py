
from utils import read_image
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import numpy as np

four_neighbors_kernel = [[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]]

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
    :param img: this equals the mask float np 2d array
    :return: 2d np array equals |N_p| in the pixel (2d-coords) p
    """
    S_without_omega = 1.0-img  # assuming img is a binary mask
    return convolve(S_without_omega, four_neighbors_kernel, mode='nearest')


def seamless_cloning(source, target, mask, offset=None, gradient_field_source_only=True):
    """
    :param source:
    :param target:
    :param mask:
    :param offset:
    :param gradient_field_source_only:
    :return:
    """
    Np = get_4_neigbours_amount(target)  # for the calc of left first term in equation (7)
    mask_neighbours_amount = get_4_neigbours_amount(mask)
    neigbours_in_sum = np.float_(mask_neighbours_amount > 0)  # for the calc of left second term in equation (7)
    omega_boundary = get_omega_boundary(mask)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = read_image('./external/main-1.jpg', 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
