import cv2, time
import numpy as np
from matplotlib import pyplot as plt
from edge_detection import *

#đọc vào ảnh 
img = cv2.imread('./Lenna.jpg', cv2.IMREAD_COLOR).astype('float64') #or int32 type

start_time = time.time()

#chuyển về với kiểu mức xám để có được chung một kênh màu nhằm phát hiện cạnh dễ hơn
img = np.dot(img[...,:3], [0.299, 0.587, 0.114])

#tích chập theo đạo hàm x và y
horizontal = Sobel_filter(img, 'x')     #gọi hàm từ edge_detection
vertical = Sobel_filter(img, 'y')

#tính theo hướng gradient
edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))

plt.title("Sobel Gradient")
plt.imshow(edged_img, cmap = 'gray')

end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
plt.show()