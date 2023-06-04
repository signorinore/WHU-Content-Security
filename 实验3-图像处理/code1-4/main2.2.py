import cv2
import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def calculate(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                     (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def cmp(img1, img2):
    p1 = cv2.split(img1)
    p2 = cv2.split(img2)
    delta = 0
    for i, j in zip(p1, p2):
        delta += calculate(i, j)
    delta = delta / 3
    return delta


if __name__ == "__main__":
    # p1 = 'vol.png'
    # p2 = 'vol2.jpg'
    # img1 = cv2.imread(p1)
    # img2 = cv2.imread(p2)
    res = []
    mid = 0
    file_list = os.listdir('test')
    for file in file_list:
        p1 = 'vol.png'
        p2 = 'test' + '/' + file
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        #print(file, tape(file))
        mid = cmp(img1, img2)
        if mid == 1:
            img = cv2.resize(img2, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('the same', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        res.append(mid)
    print(res)





