import cv2
import os
import numpy as np
import os.path as osp
from skimage import io
import random
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import joblib
import math

def hog_descriptor(image):

    if (image.max()-image.min()) != 0:
        image = (image - image.min()) / (image.max() - image.min())
        image *= 255
        image = image.astype(np.uint8)


    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    hog_feature = hog.compute(image)

    return hog_feature

def mynms(box_list,prob_list,threshold=0.8):
    x1 = box_list[:,0]
    y1 = box_list[:,1]
    x2 = box_list[:,2]
    y2 = box_list[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    box_result = []
    flag = []
    index = prob_list.argsort()[::-1] #想要从大到小排序
    while index.size>0:
        i = index[0]
        flag.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        #idx = np.where(ious < threshold)[0]
        index = np.delete(index,np.concatenate(([0],np.where(ious < threshold)[0])))
        #index = index[idx + 1]

    return box_list[flag].astype("int")


#首先通过Train中的标准的box大小计算出适合的box大小
box_list = []
ANN = os.listdir('D:/chenxi/files2/pedestrian/INRIADATA/original_images/train/annotations')
flag = "(Xmax, Ymax)"
for i in range(len(ANN)):
    for line in open('D:/chenxi/files2/pedestrian/INRIADATA/original_images/train/annotations/'+ANN[i], encoding="GBK"):
        if flag in line:
            boxsize  = line.split(flag)
            boxsize = str(boxsize[1])
            boxsize = boxsize.replace("(", "")
            boxsize = boxsize.replace(",", "")
            boxsize = boxsize.replace(")", "")
            boxsize = boxsize.replace("-", "")
            boxsize = boxsize.replace(":", "")
            boxsize = boxsize.split()
            box = (float(boxsize[2])-float(boxsize[0]) ,float(boxsize[3])-float(boxsize[1]))
            box_list.append(box)
box_list = np.array(box_list)
minlist = np.min(box_list,axis=0)
maxlist = np.max(box_list,axis=0)

"""
print(minlist)
print(maxlist)
"""
minscale = min(minlist[0]/64,minlist[1]/128)
maxscale = min(maxlist[0]/64,maxlist[1]/128)

"""
print(minscale)
print(maxscale)
"""

minscale = math.ceil(minscale)
maxscale = math.ceil(maxscale)
print(minscale)
print(maxscale)
clf = joblib.load("D:/chenxi/files2/namodel/trained_svm.m")

imglist = os.listdir('D:/chenxi/files2/pedestrian/INRIADATA/original_images/test/pos')
for i in range(101,len(imglist)):
    img = io.imread(osp.join('D:/chenxi/files2/pedestrian/INRIADATA/original_images/test/pos', imglist[i]))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    h,w,c= img.shape
    patch_list = []
    hog_feature = []
    box_list = []
    for j in range(minscale,maxscale+1,1):
        winsize = [j*64,j*128]
        for m in range(0,h-winsize[1],20):
            for n in range(0,w-winsize[0],20):
                patch = img[m:m+winsize[1],n:n+winsize[0]]
                patch = cv2.resize(patch, (64,128), interpolation = cv2.INTER_NEAREST)
                boxcoord = (m,n,m+winsize[1],n+winsize[0])
                hogfea = hog_descriptor(patch)
                hog_feature.append(hogfea)
                box_list.append(boxcoord)
                patch_list.append(patch)
    hog_feature = np.array(hog_feature).squeeze()
    box_list = np.array(box_list)
    prob = clf.predict_proba(hog_feature)[:, 1]
    mask = (prob>= 0.99)
    box_list = box_list[mask]
    prob = prob[mask]
    boxzhong = mynms(box_list,prob)

    for k in range(len(boxzhong)):
        cv2.rectangle(img, (boxzhong[k][1], boxzhong[k][0]), (boxzhong[k][3], boxzhong[k][2]),(0, 0, 255),3)
    cv2.imwrite('D:/chenxi/files2/namodel/result/'+imglist[i]+".jpg",img)
    print(str(i))
    if i == 150:
        break

print("结束")
