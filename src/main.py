import cv2
import matplotlib.pyplot as plt
import time
import config
from detect import detect_faces


# opencv loads an image into BGR color space by default
def read_img(imgpath):
    return cv2.imread(imgpath)


if config.test:
    detect_faces(read_img('data/test/test2.jpg'), 1.1, True)
