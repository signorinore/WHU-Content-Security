import os
import cv2
import pickle
import numpy as np
import numpy.ma
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont


# 查看DeepFace包含的特征点
def Print_DeepFace():
    print(dir(DeepFace))


#验证两个人是否为同一个人
def VerifyPerson():
    img_path1 = 'pics/test.png'
    # img_path2 = 'pics/test2.png'  #  相似的人脸
    img_path2 = 'pics/test3.png'  # 不相似的人脸

    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    #验证是否相似
    # 可以使用的模型
    #model_name (string): VGG-Face,
    # Facenet, OpenFace, DeepFace, DeepID,Dlib, ArcFace or Ensemble
    compare=DeepFace.verify(img1_path=img1,img2_path=img2,model_name='VGG-Face')
    print(compare)
    print('是否相似: {}'.format(compare['verified']))

    img1=cv2.resize(src=img1,dsize=(450,450))
    img2=cv2.resize(src=img2,dsize=(450,450))
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


weights='./weights/facial_expression_model_weights.h5'

#对人的特征进行检测
def FaceEmotion(img_path='pics/test.png'):
    img = cv2.imread(img_path)

    emotion = DeepFace.analyze(img_path=img_path)

    if emotion[0]['gender']['Woman'] > emotion[0]['gender']['Man']:
        str_gender = 'Woman'
    else:
        str_gender = 'Man'

    print("gender:", str_gender)
    print("age:", emotion[0]["age"])
    print("dominant_race:", emotion[0]["dominant_race"])
    print("dominant_emotion:", emotion[0]["dominant_emotion"])


    words = 'gender: ' + str_gender + '\n' \
                                              "age: " + str(emotion[0]["age"]) + '\n' \
                                                "dominant_race: " + str(emotion[0]["dominant_race"]) + '\n' \
                                            "dominant_emotion: " + str(emotion[0]["dominant_emotion"])

    img = cv2.resize(src=img, dsize=(450, 450))

    x0=emotion[0]['region']['x']
    y0=emotion[0]['region']['y']
    w=emotion[0]['region']['w']
    h=emotion[0]['region']['h']
    cv2.rectangle(img=img,pt1=(x0,y0),pt2=(x0+w,y0+h),color=(0,255,0),thickness=2)

    img_ptl = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    fillColor = (255, 0, 0)
    position = (100, 50)
    font = ImageFont.truetype('consola.ttf', 18)

    draw = ImageDraw.Draw(img_ptl)
    draw.text(position, words, font=font, fill=fillColor)
    img = cv2.cvtColor(numpy.ma.asarray(img_ptl), cv2.COLOR_BGR2RGB)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Print_DeepFace()
    # VerifyPerson()
    FaceEmotion()
