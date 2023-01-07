import cv2, time
import numpy as np
from matplotlib import pyplot as plt 
from edge_detection import *

#đọc vào ảnh 
img = cv2.imread('./Lenna.jpg',cv2.IMREAD_COLOR).astype('float64')

start_time = time.time()

#chuyển về với kiểu mức xám để có được chung một kênh màu nhằm phát hiện cạnh dễ hơn
img = np.dot(img[...,:3], [0.299, 0.587, 0.114])

#tích chập theo laplace
laplace_nega = Laplace_filter(img, 'negative')      #gọi hàm từ edge_detection
laplace = Laplace_filter(img, 'normal')

plt.title("Gaussian_filter")
plt.imshow(img, cmap = 'gray')

#lưu ảnh đã phát hiện biên cạnh
cv2.imwrite('output1_laplace.jpg', laplace)
cv2.imwrite('output2_nega_laplace.jpg', laplace_nega)

end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
plt.show()
