from PIL import Image
from pylab import *
from pcv.localdescriptors import sift
from pcv.localdescriptors import harris
import scipy

from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:/windows/fonts/SimSun.ttc", size=14)

imname = 'vol.png'
# 读入文件
im = array(Image.open(imname).convert('L'))
# 将文件转换为pmg格式
sift.process_image(imname, 'empire.sift')
# 特征提取
l1, d1 = sift.read_features_from_file('empire.sift')

figure()
gray()
subplot(121)
sift.plot_features(im, l1, circle=False)
title(u'SIFT*')

# 检测harris角点
harrisim = harris.compute_harris_response(im)

subplot(122)
filtered_coords = harris.get_harris_points(harrisim, 6, 0.1)
imshow(im)
plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
axis('off')
title(u'Harris jiaodian')

show()
