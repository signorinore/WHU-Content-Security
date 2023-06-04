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

#对数据进行处理，提取正负样本的Hog特征
#计算传入图像的HOG特征


def hog_descriptor(image):
    if (image.max()-image.min()) != 0:
        image = (image - image.min()) / (image.max() - image.min())
        image *= 255
        image = image.astype(np.uint8)
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    hog_feature = hog.compute(image)

    return hog_feature


#导入图像
poslist = os.listdir('D:/chenxi/files2/pedestrian/INRIADATA/normalized_images/train/pos')
neglist = os.listdir('D:/chenxi/files2/pedestrian/INRIADATA/normalized_images/train/neg')
testlist = os.listdir('D:/chenxi/files2/pedestrian/INRIADATA/normalized_images/test/pos')
testnlist = os.listdir('D:/chenxi/files2/pedestrian/INRIADATA/original_images/test/neg')
#获得正样本和负样本的HOG特征，并标记

hog_list = []
label_list = []
print("正样本图像有"+str(len(poslist)))
print("负样本原始图像有"+str(len(neglist))+"，每个原始图像提供十个负样本")
for i in range(len(poslist)):
    posimg = io.imread(osp.join('D:/chenxi/files2/pedestrian/INRIADATA/normalized_images/train/pos',poslist[i]))
    posimg = cv2.cvtColor(posimg,cv2.COLOR_RGBA2BGR)
    #所用图像已经经过标准化
    posimg = cv2.resize(posimg, (64, 128), interpolation=cv2.INTER_NEAREST)
    pos_hog = hog_descriptor(posimg)
    hog_list.append(pos_hog)
    label_list.append(1)
for i in range(len(neglist)):
    negimg = io.imread(osp.join('D:/chenxi/files2/pedestrian/INRIADATA/normalized_images/train/neg',neglist[i]))
    negimg = cv2.cvtColor(negimg, cv2.COLOR_RGBA2BGR)

    #在每张negimg图像中截取10张标准大小的图片作为负样本
    for j in range(10):
        y = int(random.random() * (negimg.shape[0] - 128))
        x = int(random.random() * (negimg.shape[1] - 64))
        negimgs = negimg[y:y + 128, x:x + 64]
        negimgs = cv2.resize(negimgs, (64, 128), interpolation=cv2.INTER_NEAREST)
        neg_hog = hog_descriptor(negimgs)
        hog_list.append(neg_hog)
        label_list.append(0)
print(type(hog_list[10]))
print(type(hog_list[-10]))
hog_list = np.float32(hog_list)
label_list = np.int32(label_list).reshape(len(label_list),1)

#训练SVM，并在Test上测试
clf = SVC(C=1.0, gamma='auto', kernel='rbf', probability=True)
clf.fit(hog_list.squeeze(), label_list.squeeze())
joblib.dump(clf, "D:/chenxi/files2/namodel/trained_svm.m")


#提取训练集样本和标签
test_hog = []
test_label = []
for i in range(len(testlist)):
    testimg = io.imread(osp.join('D:/chenxi/files2/pedestrian/INRIADATA/normalized_images/test/pos', testlist[i]))
    testimg = cv2.cvtColor(testimg, cv2.COLOR_RGBA2BGR)
    testimg = cv2.resize(testimg, (64, 128), interpolation=cv2.INTER_NEAREST)
    testhog = hog_descriptor(testimg)
    test_hog.append(testhog)
    test_label.append(1)

for i in range(len(testnlist)):
    testnegimg = io.imread(osp.join('D:/chenxi/files2/pedestrian/INRIADATA/original_images/test/neg',testnlist[i]))
    testnegimg = cv2.cvtColor(testnegimg, cv2.COLOR_RGBA2BGR)

    #在每张negimg图像中截取10张标准大小的图片作为负样本
    for j in range(10):
        y = int(random.random() * (testnegimg.shape[0] - 128))
        x = int(random.random() * (testnegimg.shape[1] - 64))
        testnegimgs = testnegimg[y:y + 128, x:x + 64]
        testnegimgs = cv2.resize(testnegimgs, (64, 128), interpolation=cv2.INTER_NEAREST)
        testneg_hog = hog_descriptor(testnegimgs)
        test_hog.append(testneg_hog)
        test_label.append(0)
test_hog = np.float32(test_hog)
test_label = np.int32(test_label).reshape(len(test_label),1)
#可以导入训练后的SVM
clf = joblib.load("D:/chenxi/files2/namodel/trained_svm.m")


#对训练集进行预测并绘制PR、ROC曲线计算AUC值
prob = clf.predict_proba(test_hog.squeeze())[:, 1]

precision, recall, thresholds_1 = metrics.precision_recall_curve(test_label.squeeze(), prob)

plt.figure(figsize=(20, 20), dpi=100)
plt.plot(precision, recall, c='red')
plt.scatter(precision, recall, c='blue')
plt.xlabel("precision", fontdict={'size': 16})
plt.ylabel("recall", fontdict={'size': 16})
plt.title("PR_curve", fontdict={'size': 20})
plt.savefig('D:/chenxi/files2/namodel/savefig/PR.png',dpi=300)
Ap=metrics.average_precision_score(test_label.squeeze(), prob)

fpr, tpr, thresholds_2 = metrics.roc_curve(test_label.squeeze(), prob, pos_label=1)

plt.figure(figsize=(20, 20), dpi=100)
plt.plot(fpr, tpr, c='red')
plt.scatter(fpr, tpr, c='blue')
plt.xlabel("FPR", fontdict={'size': 16})
plt.ylabel("TPR", fontdict={'size': 16})
plt.title("ROC_curve", fontdict={'size': 20})
plt.savefig('D:/chenxi/files2/namodel/savefig/ROC.png', dpi=300)

AUC=metrics.roc_auc_score(test_label.squeeze(), prob)
print(AUC)
print(Ap)
