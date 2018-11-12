import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.misc as smsc

from PIL import Image

img = Image.open('cont.jpg')
width, height = img.size
print(width, height)
img.show()

# img = cv2.imread('cont.jpg')
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()