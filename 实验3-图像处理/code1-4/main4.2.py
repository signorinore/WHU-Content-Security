from PIL import Image
from pylab import *
from pcv.localdescriptors import sift
import matplotlib.pyplot as plt # plt 用于显示图片


im1f = 'pics/3.jpg'
im1 = array(Image.open(im1f))
sift.process_image(im1f, 'pics_res/out_sift_1.txt')
l1, d1 = sift.read_features_from_file('pics_res/out_sift_1.txt')

arr=[]#单维链表数组
arrHash = {}#字典型数组
for i in range(1, 12):
    im2f = 'pics/'+str(i)+'.jpg'
    im2 = array(Image.open(im2f))
    sift.process_image(im2f, 'pics_res/out_sift_2.txt')
    l2, d2 = sift.read_features_from_file('pics_res/out_sift_2.txt')
    matches = sift.match_twosided(d1, d2)
    length=len(matches.nonzero()[0])
    length=int(length)
    arr.append(length)#添加新的值
    arrHash[length]=im2f#添加新的值

arr.sort()#数组排序
arr=arr[::-1]#数组反转
arr=arr[:3]#截取数组元素到第3个
i=0
plt.figure(figsize=(6,12))#设置输出图像的大小
for item in arr:
    if(arrHash.get(item)!=None):
        img=arrHash.get(item)
        im1 = array(Image.open(img))
        ax=plt.subplot(511 + i)#设置子团位置
        ax.set_title('{} matches'.format(item))#设置子图标题
        plt.axis('off')#不显示坐标轴
        imshow(im1)
        i = i + 1

plt.show()
