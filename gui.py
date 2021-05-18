import argparse
import numpy as np
import cv2
from os import path



class MaskPainter:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image_copy = self.image.copy()

        self.mask = np.zeros(self.image.shape)
        self.mask_copy = self.mask.copy()
        self.size = 4
        self.to_draw = False

        self.window_name = "Draw mask. s:save; r:reset; q:quit"

    def _paint_mask_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.to_draw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_draw:
                cv2.rectangle(self.image, (x - self.size, y - self.size),
                              (x + self.size, y + self.size),
                              (0, 255, 0), -1)
                cv2.rectangle(self.mask, (x - self.size, y - self.size),
                              (x + self.size, y + self.size),
                              (255, 255, 255), -1)
                cv2.imshow(self.window_name, self.image)

        elif event == cv2.EVENT_LBUTTONUP:
            self.to_draw = False

    def paint_mask(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,
                             self._paint_mask_handler)

        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                self.image = self.image_copy.copy()
                self.mask = self.mask_copy.copy()

            elif key == ord("s"):
                break

            elif key == ord("f"):
                break

            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        maskPath = path.join(path.dirname(self.image_path),
                             'mask.png')
        cv2.imwrite(maskPath, self.mask)

        # close all open windows
        cv2.destroyAllWindows()
        return maskPath


class MaskMover:

    def __init__(self, image_path, mask_path):
        self.image_path, self.mask_path = image_path, mask_path
        self.image = cv2.imread(image_path)
        self.image_copy = self.image.copy()

        self.original_mask = cv2.imread(mask_path)
        self.original_mask_copy = np.zeros(self.image.shape)
        self.original_mask_copy[np.where(self.original_mask != 0)] = 255

        self.mask = self.original_mask_copy.copy()

        self.to_move = False
        self.x0 = 0
        self.y0 = 0
        self.is_first = True
        self.xi = 0
        self.yi = 0

        self.window_name = "Move the mask. s:save; r:reset; q:quit"

    def _blend(self, image, mask):
        ret = image.copy()
        alpha = 0.3
        ret[mask != 0] = ret[mask != 0] * alpha + 255 * (1 - alpha)
        return ret.astype(np.uint8)

    def _move_mask_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.to_move = True
            if self.is_first:
                self.x0, self.y0 = x, y
                self.is_first = False

            self.xi, self.yi = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_move:
                M = np.float32([[1, 0, x - self.xi],
                                [0, 1, y - self.yi]])
                self.mask = cv2.warpAffine(self.mask, M,
                                           (self.mask.shape[1],
                                            self.mask.shape[0]))
                cv2.imshow(self.window_name,
                           self._blend(self.image, self.mask))
                self.xi, self.yi = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.to_move = False

    def move_mask(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,
                             self._move_mask_handler)

        while True:
            cv2.imshow(self.window_name,
                       self._blend(self.image, self.mask))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                self.image = self.image_copy.copy()
                self.mask = self.original_mask_copy.copy()

            elif key == ord("s"):
                break

            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        new_mask_path = path.join(path.dirname(self.image_path),
                                  'target_mask.png')
        cv2.imwrite(new_mask_path, self.mask)

        # close all open windows
        cv2.destroyAllWindows()
        return self.xi - self.x0, self.yi - self.y0, new_mask_path

