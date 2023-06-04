# -*- coding: utf-8 -*-
import time

import matplotlib.pyplot as plt
import cv2
import numpy as np


# 预处理
import torch


def imgProcess(path):
    img = cv2.imread(path)
    # print(type(img))
    # img = img[:,:,::-1]
    # 统一规定大小
    img = cv2.resize(img, (640, 480))
    # 高斯模糊
    img_Gas = cv2.GaussianBlur(img, (5, 5), 0)
    # RGB通道分离
    img_B = img[:, :, 0]
    img_G = img[:, :, 1]
    img_R = img[:, :, 2]
    # 读取灰度图和HSV空间图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, img_Gas, img_B, img_G, img_R, img_gray, img_HSV


# 初步识别
def preIdentification(img_gray, img_HSV, img_B, img_R):
    img_w = img_gray.shape[0]
    img_h = img_gray.shape[1]
    for i in range(img_w):
        for j in range(img_h):
            # 普通蓝色车牌，同时排除透明反光物质的干扰
            if (img_HSV[:, :, 0][i, j] < 180) and (img_B[i, j] > 20) and (img_R[i, j] < 100):
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0
    #
    # plt.imshow(img_gray,cmap='gray')
    # plt.show()

    # 定义核
    kernel_small = np.ones((3, 3))
    kernel_big = np.ones((7, 7))

    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 高斯平滑
    img_di = cv2.dilate(img_gray, kernel_small, iterations=5)  # 腐蚀5次
    img_close = cv2.morphologyEx(img_di, cv2.MORPH_CLOSE, kernel_big)  # 闭操作
    img_close = cv2.GaussianBlur(img_close, (5, 5), 0)  # 高斯平滑
    _, img_bin = cv2.threshold(img_close, 100, 255, cv2.THRESH_BINARY)  # 二值化
    return img_bin


def verifySizes(RotatedRect):
    RotatedRect = cv2.minAreaRect(RotatedRect)
    error = 0.4
    aspect = 4.7272
    min = 15 * aspect * 15
    max = 125 * aspect * 125
    rmin = aspect - aspect * error
    rmax = aspect + aspect * error
    height, width = RotatedRect[1]
    if height == 0 or width == 0:
        return False
    area = height * width
    r = width / height
    if r < 1:
        r = height / width
    if (area < min or area > max) or (r < rmin or r > rmax):
        return False
    else:
        return True


# 定位
def fixPosition(img, img_bin):
    # 检测所有外轮廓，只留矩形的四个顶点
    contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # img_draw = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
    # plt.imshow(img_draw)
    # plt.show()

    # print(len(contours))
    # 形状及大小筛选校验
    det_x_max = 0
    det_y_max = 0
    num = 0
    for i in range(len(contours)):
        # print(contours[i])
        # print("------------------")

        x_min = np.min(contours[i][:, :, 0])
        x_max = np.max(contours[i][:, :, 0])
        y_min = np.min(contours[i][:, :, 1])
        y_max = np.max(contours[i][:, :, 1])
        det_x = x_max - x_min
        det_y = y_max - y_min
        if (det_x > det_x_max) and (det_y > det_y_max):
            det_y_max = det_y
            det_x_max = det_x
            num = i
    # 获取最可疑区域轮廓点集
    points = np.array(contours[num][:, 0])
    # print(points)
    return points


# img_lic_canny = cv2.Canny(img_lic_bin, 100, 200)


def findVertices(points):
    # 获取最小外接矩阵，中心点坐标，宽高，旋转角度
    rect = cv2.minAreaRect(points)
    # 获取矩形四个顶点，浮点型
    box = cv2.boxPoints(rect)
    # 取整
    box = np.int0(box)
    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    # 上下左右四个点坐标
    vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y],
                         [right_point_x, right_point_y]])

    return vertices, rect


def tiltCorrection(vertices, rect):
    # 畸变情况1
    # print(rect)
    if rect[2] == 90.0:
        new_right_point_x = vertices[0, 0]
        new_right_point_y = vertices[1, 1]
        new_left_point_x = vertices[1, 0]
        new_left_point_y = vertices[0, 1]
        point_set_1 = np.float32([[0, 0], [94, 0], [94, 24], [0, 24]])
        # new_box = np.array([(vertices[0, 0], vertices[0, 1]), (new_left_point_x, new_left_point_y), (vertices[1, 0], vertices[1, 1]),
        #   (new_right_point_x, new_right_point_y)])

    elif rect[2] > 45:
        new_right_point_x = vertices[0, 0]
        if vertices[3, 0] == vertices[1, 0]:
            new_right_point_y = vertices[1, 1]
        else:
            new_right_point_y = int(
                vertices[1, 1] - (vertices[0, 0] - vertices[1, 0]) / (vertices[3, 0] - vertices[1, 0]) * (
                        vertices[1, 1] - vertices[3, 1]))

        new_left_point_x = vertices[1, 0]
        if vertices[0, 0] == vertices[2, 0]:
            new_left_point_y = vertices[0, 1]
        else:
            new_left_point_y = int(
                vertices[0, 1] + (vertices[0, 0] - vertices[1, 0]) / (vertices[0, 0] - vertices[2, 0]) * (
                        vertices[2, 1] - vertices[0, 1]))
        # 校正后的四个顶点坐标
        point_set_1 = np.float32([[94, 0], [0, 0], [0, 24], [94, 24]])
        # new_box = np.array([(vertices[0, 0], vertices[0, 1]), (vertices[1, 0], vertices[0, 1]), (vertices[1, 0], vertices[1, 1]),
        #   ( vertices[0, 0], vertices[1, 1])])

    # 畸变情况2
    elif rect[2] <= 45:
        new_right_point_x = vertices[1, 0]
        if vertices[3, 0] == vertices[0, 0]:
            new_right_point_y = vertices[0, 1]
        else:
            new_right_point_y = int(vertices[0, 1] + (vertices[1, 0] - vertices[0, 0]) / (vertices[3, 0] - vertices[0, 0]) * (
                    vertices[3, 1] - vertices[0, 1]))
        new_left_point_x = vertices[0, 0]
        if vertices[1, 0] == vertices[2, 0]:
            new_left_point_y = vertices[1, 1]
        else :
            new_left_point_y = int(vertices[1, 1] - (vertices[1, 0] - vertices[0, 0]) / (vertices[1, 0] - vertices[2, 0]) * (
                    vertices[1, 1] - vertices[2, 1]))
        # 校正后的四个顶点坐标
        point_set_1 = np.float32([[0, 0], [0, 24], [94, 24], [94, 0]])
        # new_box = np.array([(vertices[0, 0], vertices[0, 1]), (vertices[0, 0], vertices[1, 1]), (vertices[1, 0], vertices[1, 1]),
        #   (vertices[1, 0], vertices[0, 1])])

    # 校正前平行四边形四个顶点坐标
    new_box = np.array([(vertices[0, 0], vertices[0, 1]), (new_left_point_x, new_left_point_y), (vertices[1, 0], vertices[1, 1]),
      (new_right_point_x, new_right_point_y)])

    point_set_0 = np.float32(new_box)
    return point_set_0, point_set_1, new_box


def transform(img, point_set_0, point_set_1):
    # 变换矩阵
    mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
    # 投影变换
    lic = cv2.warpPerspective(img, mat, (94, 24))
    return lic


def get_BlueImg_bin(img):
    # 掩膜：BGR通道，若像素B分量在 100~255 且 G分量在 0~190 且 G分量在 0~140 置255（白色） ，否则置0（黑色）
    mask_gbr = cv2.inRange(img, (100, 0, 0), (255, 190, 140))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换成 HSV 颜色空间
    h, s, v = cv2.split(img_hsv)  # 分离通道  色调(H)，饱和度(S)，明度(V)
    mask_s = cv2.inRange(s, 80, 255)  # 取饱和度通道进行掩膜得到二值图像
    rgbs = mask_gbr & mask_s  # 与操作，两个二值图像都为白色才保留，否则置黑
    # 第二个参数是核的横向，核的横向分量大，使车牌数字尽量连在一起
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))
    img_rgbs_dilate = cv2.dilate(rgbs, kernel, 3)  # 膨胀 ，减小车牌空洞
    return img_rgbs_dilate


def get_EdgeImg_bin(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobelx = np.uint8(np.absolute(sobelx))
    # kernel 横向分量大，使车牌数字尽量连在一起
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 3))
    closing_img = cv2.morphologyEx(sobelx, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算
    return cv2.inRange(closing_img, 170, 255)


def get_other_image(img):
    gray_car = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step2  高斯模糊处理
    blur_car = cv2.GaussianBlur(gray_car, (5, 5), 0)

    # Step3  Sobel计算水平导数
    sobel_car = cv2.Sobel(blur_car, cv2.CV_16S, 1, 0)
    sobel_car = cv2.convertScaleAbs(sobel_car)  # 转回uint8

    # Step4  Otsu大津算法自适应阈值二值化
    _, otsu_car = cv2.threshold(sobel_car, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    # Step5  闭操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    close_car = cv2.morphologyEx(otsu_car, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('sss',close_car)
    # Step6  提取外部轮廓

    return close_car
    # img, contours, hierarchy = cv2.findContours(close_car, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


def is_change_valid(point):
    min_x = -100
    min_y = -60

    if point[0][0] < min_x or point[1][0] < min_x or point[2][0] < min_x or point[3][0] < min_x:
        return False
    if point[0][1] < min_y or point[1][1] < min_y or point[2][1] < min_y or point[3][1] < min_y:
        return False

    # if point[0][0] > 640 or point[0][1] < 0 or point[1][0] < 0 or point[1][1] < 0:
    #     return False

    if (abs(point[0][0] - point[2][0]) < 500 or abs(point[0][1] - point[2][1]) < 300) \
            and (abs(point[1][0] - point[3][0]) < 500 or abs(point[1][1] - point[3][1]) < 300):
        return False

    return True


def main():


    start_time = time.time()
    for i in range(1000):
        path = './demo/raw/028.jpg'
        # 图像预处理
        img, img_Gas, img_B, img_G, img_R, img_gray, img_HSV = imgProcess(path)

        img_tensor = torch.from_numpy(img)
        img = img_tensor.numpy()
        # print(img)
        # print("---------")
        # print(img)

        img_other = get_BlueImg_bin(img)

        # plt.imshow(img_other, cmap="gray")
        # plt.show()

        points = fixPosition(img, img_other)
        vertices, rect = findVertices(points)

        # print(vertices)
        # img_draw = cv2.drawContours(img.copy(), [vertices], -1, (0, 0, 255), 3)
        # plt.imshow(img_draw)
        # plt.show()

        point_set_0, point_set_1, new_box = tiltCorrection(vertices, rect)

        # print(point_set_0)
        # print(point_set_1)

        if is_change_valid(point_set_0):
            lic = transform(img, point_set_0, point_set_1)
        else:
            lic = cv2.imread(path)

        # plt.subplot(1, 3, 1)
        # plt.imshow(cv2.imread(path)[:, :, ::-1])
        # plt.subplot(1, 3, 2)
        # img_draw = cv2.drawContours(img.copy(), [new_box], -1, (0, 0, 255), 3)
        # plt.imshow(img_draw[:, :, ::-1])
        # plt.subplot(1, 3, 3)
        # plt.imshow(lic[:, :, ::-1])
        # plt.show()

        # print(type(lic))


    print(time.time() - start_time)




if __name__ == '__main__':
    main()
