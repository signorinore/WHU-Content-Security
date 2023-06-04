### 文档说明

#### pedestrain

code1-4/main3、main3.2的数据集，里面有很多行人的图片。

#### code1-4

main1.py：实现图片 titanic.jpg 的仿射变换

main2.py：完成了提取 vol.png 的直方图特征

main2.2.py：实现了根据直方图特征来匹配查找图片。test文件夹有5张图片，此程序完成了在test文件夹中，根据直方图特征找到与 vol.png 最相似的一张图片。

main3.py：提取图片的HOG特征。

main3.2.py：基于HOG特征和SVM分类器的行人检测模型的训练和测试代码。

main4.py：提取 vol.png 的SIFT特征和Harris角点特征。

main4.2.py：通过SIFT特征，识别一个文件夹中与目标图片最相似的三张图片。pics文件夹有12张图片；这里将pics/3.jpg作为目标图片，该程序实现在pics中寻找三张与3.jpg最相似的3张图片。pics_res文件夹是运行过程中输出的过程文件。

#### code5

此任务需要在虚拟环境中实现。这里使用的是Ubuntu22.04-x64。部署了yolov5模型并完成图像目标检测任务。具体过程见实验报告。