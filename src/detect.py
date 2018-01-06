import numpy as np
import cv2


# load HAAR cascade classifier training file
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')


def convert2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def convert2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_faces(img, scale_factor=1.1, show=False):
    img_copy = np.copy(img)
    gray = convert2gray(img_copy)

    # detect multiscale (some images may be closer to camera than others) images
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=5);

    # loop and draw them as rectangles on the original colored image
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if show:
        cv2.imshow('Image', img_copy)
        cv2.waitKey(0)

    return img_copy