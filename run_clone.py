import cv2
import getopt
import sys
from gui import MaskPainter, MaskMover
from clone import seamless_cloning, shepards_seamless_cloning
from utils import read_image, plt
from os import path


def usage():
    print(
        "Usage: python run_clone.py [options] \n\n\
        Options: \n\
        \t-h\t Flag to specify a brief help message and exits..\n\
        \t-s\t(Required) Specify a source image.\n\
        \t-t\t(Required) Specify a target image.\n\
        \t-m\t(Optional) Specify a mask image with the object in white and other part in black, ignore this option if you plan to draw it later.\n\
        \t-x\t(Optional) Flag to specify a mode, either 'possion' or 'shepard'. default is possion.\n\
        \t-v\t(Optional) Flag to specify grad field of source only or both in case of Possion solver is used. default is source only.")


if __name__ == '__main__':
    # parse command line arguments
    args = {}

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "vxhs:t:m:p:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        print("See help: run_clone.py -h")
        exit(2)
    for o, a in opts:
        if o in ("-h"):
            usage()
            exit()
        elif o in ("-s"):
            args["source"] = a
        elif o in ("-t"):
            args["target"] = a
        elif o in ("-m"):
            args["mask"] = a
        elif o in ("-x"):
            args["mode"] = a.lower()
        elif o in ("-v"):
            args["gradient_field_source_only"] = a
        else:
            continue

    #
    if ("source" not in args) or ("target" not in args):
        usage()
        exit()

    #
    # set default mode to Possion solver
    mode = "poisson" if ("mode" not in args) else args["mode"]
    gradient_field_source_only = ("gradient_field_source_only" not in args)

    source = read_image(args["source"], 2)
    target = read_image(args["target"], 2)

    if source is None or target is None:
        print('Source or target image not exist.')
        exit()

    if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
        print('Source image cannot be larger than target image.')
        exit()

    # draw the mask
    mask_path = ""
    if "mask" not in args:
        print('Please highlight the object to disapparate.\n')
        mp = MaskPainter(args["source"])
        mask_path = mp.paint_mask()
    else:
        mask_path = args["mask"]

    # adjust mask position for target image
    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(args["target"], mask_path)
    offset_x, offset_y, target_mask_path = mm.move_mask()

    # blend
    print('Blending ...')
    target_mask = read_image(target_mask_path, 1)
    offset = offset_x, offset_y

    cloning_tool = seamless_cloning if mode == "poisson" else shepards_seamless_cloning
    kwargs = {"gradient_field_source_only": gradient_field_source_only} if mode == "poisson" else {}
    blend_result = cloning_tool(source, target, target_mask, offset, **kwargs)

    cv2.imwrite(path.join(path.dirname(args["source"]), 'target_result.png'),
                blend_result)
    plt.figure("Result"), plt.imshow(blend_result), plt.show()
    print('Done.\n')


'''
running example:

- Possion based solver:
python run_clone.py -s external/blend-1.jpg -t external/main-1.jpg
python run_clone.py -s external/source3.jpg -t external/target3.jpg -v

- Shepard's interpolation:
python run_clone.py -s external/blend-1.jpg -t external/main-1.jpg -x 
python run_clone.py -s external/source3.jpg -t external/target3.jpg -x
'''