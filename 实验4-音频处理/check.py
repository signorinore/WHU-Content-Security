import pandas as pd
import os
from scipy.io import wavfile
import struct
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
import progressbar
import numpy as np

data = pd.read_csv('UrbanSound8K.csv')

#  查看数据集前5行——————————————————————
# content = data.head()
# print(content)
#  ———————————————————————————————————

#  查看各个文件夹中的声音分布情况——————————
# appended = []
# for i in range(1, 11):
#     appended.append(data[data.fold == i]['class'].value_counts())
#
# class_distribution = pd.DataFrame(appended)
# class_distribution = class_distribution.reset_index()
# class_distribution['index'] = ["fold" + str(x) for x in range(1, 11)]
# print(class_distribution)
#  ————————————————————————————————————


# 读取wav文件函数
def path_class(filename):
    excerpt = data[data['slice_file_name'] == filename]
    path_name = os.path.join('D:/chenxi/files2/UrbanSound8K', 'fold'+str(excerpt.fold.values[0]), filename)
    return path_name, excerpt['class'].values[0]


# 绘图wav函数
def wav_plotter(full_path, class_label):
    rate, wav_sample = wavfile.read(full_path)
    wave_file = open(full_path, "rb")
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    print('sampling rate: ',rate,'Hz')
    print('bit depth: ',bit_depth)
    print('number of channels: ',wav_sample.shape[1])
    print('duration: ',wav_sample.shape[0]/rate,' second')
    print('number of samples: ',len(wav_sample))
    print('class: ',class_label)
    plt.figure(figsize=(12, 4))
    plt.plot(wav_sample)
    print("SUCCESS")
    plt.show()
    return ipd.Audio(full_path)


if __name__ == "__main__":
    #  展示单个音频信息以及可视化图片
    fullpath, label = path_class('100263-2-0-117.wav')
    wav_plotter(fullpath, label)

