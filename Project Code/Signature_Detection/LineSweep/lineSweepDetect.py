from PIL import Image

import os
import numpy as np
import cv2

from itertools import product
from ufarray import *


def main():
    images_dir = "cheque_images"
    input_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), images_dir
    )

    for filename in os.listdir(input_path):
        fileSize = os.stat(os.path.join(input_path, filename)).st_size

        if fileSize != 0:
            print(filename)
            img = Image.open(os.path.join(input_path,filename))
            temp = np.array(img)

            grayscale = img.convert("L")
            xtra, thresh = cv2.threshold(
                np.array(grayscale), 128, 255, cv2.THRESH_BINARY_INV
            )

            # thresh = cv2.medianBlur(thresh, 2)

            rows = thresh.shape[0]
            cols = thresh.shape[1]

            flagx = 0
            indexStartX = 0
            indexEndX = 0

            for i in range(rows):
                line = thresh[i, :]

                if flagx == 0:
                    ele = [255]
                    mask = np.isin(ele, line)

                    if True in mask:
                        indexStartX = i
                        flagx = 1
                        # print('start x: ', indexStartX, flagx)

                elif flagx == 1:
                    ele = [255]
                    mask = np.isin(ele, line)

                    if True in mask:
                        indexEndX = i
                        # print('end x: ', indexEndX)
                    elif indexStartX + 5 > indexEndX:
                        indexStartX = 0
                        flagx = 0
                        # print('elif x: ', indexStartX, flagx)
                    else:
                        break

            flagy = 0
            indexStartY = 0
            indexEndY = 0

            for j in range(cols):
                line = thresh[indexStartX:indexEndX, j : j + 20]

                if flagy == 0:
                    ele = [255]
                    mask = np.isin(ele, line)

                    if True in mask:
                        indexStartY = j
                        flagy = 1
                        # print('start y: ', indexStartY, flagy)

                elif flagy == 1:
                    ele = [255]
                    mask = np.isin(ele, line)

                    if True in mask:
                        indexEndY = j
                        # print('end y: ', indexEndY)
                    elif indexStartY + 20 > indexEndY:
                        indexStartY = 0
                        flagy = 0
                        # print('elif y: ', indexStartY, flagy)
                    else:
                        break

            # print(indexStartX, indexEndX, indexStartY, indexEndY)
            cv2.line(
                thresh,
                (indexStartY, indexStartX),
                (indexEndY, indexStartX),
                (255, 0, 0),
                1,
            )
            cv2.line(
                thresh,
                (indexStartY, indexEndX),
                (indexEndY, indexEndX),
                (255, 0, 0),
                1,
            )

            cv2.line(
                thresh,
                (indexStartY, indexStartX),
                (indexStartY, indexEndX),
                (255, 0, 0),
                1,
            )
            cv2.line(
                thresh,
                (indexEndY, indexStartX),
                (indexEndY, indexEndX),
                (255, 0, 0),
                1,
            )

            temp_np = temp[
                indexStartX : indexEndX + 1, indexStartY : indexEndY + 1
            ]

            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sigCrop")
            if not os.path.exists(path):
                os.makedirs(path)

            s1 = "Result_" + filename
            cv2.imwrite(os.path.join(path, s1), temp_np)

            # cv2.imshow('2', cv2.resize(thresh, (600, 600)))
            # cv2.waitKey(0)


if __name__ == "__main__":
    main()
