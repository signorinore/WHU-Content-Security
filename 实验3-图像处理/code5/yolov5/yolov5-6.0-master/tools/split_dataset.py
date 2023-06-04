import os
import random

import shutil
from shutil import copy2

trainfiles = os.listdir(r"C:\Code\Licence-Recognition\datasets_LPRNet\raw")  # （图片文件夹）
num_train = len(trainfiles)
print("num_train: " + str(num_train))
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)  # 打乱顺序
num = 0
trainDir = r"C:\Code\Licence-Recognition\datasets_LPRNet\images\train"
validDir = r"C:\Code\Licence-Recognition\datasets_LPRNet\images\val"
detectDir = r"C:\Code\Licence-Recognition\datasets_LPRNet\images\test"
for i in index_list:
    fileName = os.path.join(r"C:\Code\Licence-Recognition\datasets_LPRNet\raw", trainfiles[i])
    if num < num_train * 0.8:  # 8:1:1
        # print(str(fileName))
        copy2(fileName, trainDir)
    elif num < num_train * 0.9:
        # print(str(fileName))
        copy2(fileName, validDir)
    else:
        # print(str(fileName))
        copy2(fileName, detectDir)
    if num % 1000 == 0:
        print("已完成:{}".format(num))
    num += 1
