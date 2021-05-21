from clone import *
from utils import read_image, plot, pyramid_blending_example1


def shepards_blending_example1():
    target = read_image('./external/main-1.jpg', 2)
    source = read_image('./external/blend-1.jpg', 2)
    mask = read_image('./external/mask-1.jpg', 1)
    offset = (0, 0)
    mask = mask > 0.1
    F = create_mask_dist_transform(mask)
    cloned = shepards_seamless_cloning(source, target, mask, offset, F)
    plot(source, target, mask, cloned, title="Shepard's Based Blending 1")


def shepards_blending_example2():
    target = read_image('./external/target1.jpg', 2)
    source = read_image('./external/source1.jpg', 2)
    mask = read_image('./external/mask1.png', 1)
    mask = mask > 0.1
    offset = (0, 66)
    F = create_mask_dist_transform(mask)
    cloned = shepards_seamless_cloning(source, target, mask, offset, F)
    plot(source, target, mask, cloned, title="Shepard's Based Blending 2")


def shepards_blending_example3():
    target = read_image('./external/target3.jpg', 2)
    source = read_image('./external/source3.jpg', 2)
    mask = read_image('./external/mask3-drawn.jpeg', 1)
    mask = mask > 0.1
    offset = (0, 0)
    F = create_mask_dist_transform(mask)
    cloned = shepards_seamless_cloning(source, target, mask, offset, F)
    plot(source, target, mask, cloned, title="Shepard's Based Blending OMER")


def poisson_blending_example1(monochromatic_source=False):
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
    plot(source, target, mask, cloned, title='Possion Based Blending 1')


def poisson_blending_example2():
    target = read_image('./external/target1.jpg', 2)
    source = read_image('./external/source1.jpg', 2)
    mask = read_image('./external/mask1.png', 1)
    offset = (0, 66)
    cloned = seamless_cloning(source, target, mask, offset=offset)
    plot(source, target, mask, cloned, title='Possion Based Blending 2')


def poisson_blending_example3():
    target = read_image('./external/target3.jpg', 2)
    source = read_image('./external/source3.jpg', 2)
    mask = read_image('./external/mask3-drawn.jpeg', 1)
    offset = (0, 0)
    cloned = seamless_cloning(source, target, mask, offset=offset)
    plot(source, target, mask, cloned, title='Possion Based Blending 3')


def poisson_blending_example3_coarse_mask(gradient_field_source_only=False):
    target = read_image('./external/target3.jpg', 2)
    source = read_image('./external/source3.jpg', 2)
    mask = read_image('./external/mask-3-coarse.jpg', 1)
    offset = (0, 0)
    cloned = seamless_cloning(source, target, mask, offset=offset,
                              gradient_field_source_only=gradient_field_source_only)
    plot(source, target, mask, cloned,
         title='Possion Based Blending coarse 3 - grad field flag %s' % gradient_field_source_only)


def shaprds_blending_example3_coarse_mask():
    target = read_image('./external/target3.jpg', 2)
    source = read_image('./external/source3.jpg', 2)
    mask = read_image('./external/mask-3-coarse.jpg', 1)
    offset = (0, 0)
    cloned = shepards_seamless_cloning(source, target, mask, offset=offset)
    plot(source, target, mask, cloned, title='Shepards Based Blending coarse 3')


if __name__ == '__main__':
    shepards_blending_example1()
    shepards_blending_example2()
    shepards_blending_example3()
    shaprds_blending_example3_coarse_mask()

    poisson_blending_example1()
    poisson_blending_example2()
    poisson_blending_example3()
    poisson_blending_example3_coarse_mask()

    poisson_blending_example1(monochromatic_source=True)
    pyramid_blending_example1()
