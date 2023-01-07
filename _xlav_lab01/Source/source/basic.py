import numpy as np
import cv2 
from matplotlib import pyplot as plt

path = r'./Lenna.jpg'

#đọc vào ảnh
img_color = cv2.imread(path, cv2.IMREAD_COLOR)

#sắp xếp lại các kênh màu
img_cvt = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
print(img_cvt.size) #ghi thông tin của ảnh

#chuyển xám
img_gray = np.dot(img_color[...,:3], [0.299, 0.587, 0.114])
print(img_gray.shape)

cv2.line(img_cvt, (27,3), (29,70), (255,0,0), 3)
print("\n màu tại vị trí pixel [cột, hàng]: ", img_cvt[29, 29])
crop = img_cvt[100:300, 100:300]
copy = img_cvt.copy()
copy[100:300, 100:300] = (255, 0, 0) #assign blue color

#hiển thị lên màn hình, có thể dùng plt.imsave hoặc cv2.write để lưu ảnh 
plt.subplot(3,3,5), plt.title("Crop"),plt.imshow(crop[:,:,::-1]), plt.xticks([]), plt.yticks([]) 
plt.subplot(3,3,4), plt.title("Copy_image"), plt.imshow(copy[:,:,::-1]), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3), plt.title("Gray Color"), plt.imshow(img_gray, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,1), plt.title("BGR Color"), plt.imshow(img_color), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,2), plt.title("RGB Color"), plt.imshow(img_cvt), plt.xticks([]), plt.yticks([])

plt.show()

#cv2.waitKey(0)
#cv2.destroyAllWindows()



