import shutil
import cv2
import os


def txt_translate(path, txt_path):
    for filename in os.listdir(path):
        # print(filename)

        list1 = filename.split("-", 4)  # 第一次分割，以减号'-'做分割
        subname = list1[3]
        list2 = filename.split(".", 1)
        subname1 = list2[1]
        if subname1 == 'txt':
            continue
        point1, point2, point3, point4 = subname.split("_")  # 第二次分割，以下划线'_'做分割
        x1, y1 = point1.split("&", 1)
        x2, y2 = point2.split("&", 1)
        x3, y3 = point3.split("&", 1)
        x4, y4 = point4.split("&", 1)

        x_min = min(x1,x2,x3,x4)
        y_min = min(y1,y2,y3,y4)
        x_max = max(x1,x2,x3,x4)
        y_max = max(y1,y2,y3,y4)

        width = int(x_max) - int(x_min)
        height = int(y_max) - int(y_min)  # bounding box的宽和高
        cx = (float(x_min)+float(x_max))/2.0
        cy = (float(y_min)+float(y_max))/2.0  # bounding box中心点

        img = cv2.imread(path + filename)

        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)
        txtfile = txt_path + "\\" + txtname[0] + ".txt"
        # txtfile = txt_path + txtname[0] + ".txt"
        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, 'a+') as f:
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))


if __name__ == '__main__':
    # det图片存储地址
    trainDir = r"C:\Code\Licence-Recognition\datasets\images\train\\"
    validDir = r"C:\Code\Licence-Recognition\datasets\images\val\\"
    testDir = r"C:\Code\Licence-Recognition\datasets\images\test\\"
    # det txt存储地址
    train_txt_path = r"C:\Code\Licence-Recognition\datasets\labels\train"
    val_txt_path = r"C:\Code\Licence-Recognition\datasets\labels\val"
    test_txt_path = r"C:\Code\Licence-Recognition\datasets\labels\test"
    txt_translate(trainDir, train_txt_path)
    txt_translate(validDir, val_txt_path)
    txt_translate(testDir, test_txt_path)
