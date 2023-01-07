import numpy as np
import cv2, time
import matplotlib.pyplot as plt

#kiểm tra thang màu ở mỗi kênh để đảm điểm ảnh không vượt quá giới hạn của nó
def check(value):
    if (value < 0):
        return 0
    if (value > 255):
        return 255
    return value

def change_brightness(img, beta):
    img_new = np.asarray(img, dtype = int)

    for i in range(img_new.shape[0]):
        for j in range (img_new.shape[1]):
            img_new[i, j][0] = check(img_new[i, j][0] + beta)
            img_new[i, j][1] = check(img_new[i, j][1] + beta)
            img_new[i, j][2] = check(img_new[i, j][2] + beta)

    return img_new

def change_contrast(img, alpha):
    img_new = np.asarray(img, dtype = int)

    for i in range(img_new.shape[0]):
        for j in range (img_new.shape[1]):
            img_new[i, j][0] = alpha*img_new[i, j][0]
            img_new[i, j][1] = alpha*img_new[i, j][1]
            img_new[i, j][2] = alpha*img_new[i, j][2]

    return img_new

def brightness_contrast(img, alpha, beta):
    img_new = np.asarray(img, dtype = int)   # cast pixel values to int
    for i in range(img_new.shape[0]):
        for j in range (img_new.shape[1]):
            img_new[i, j][0] = check(alpha*img_new[i, j][0] + beta)
            img_new[i, j][1] = check(alpha*img_new[i, j][1] + beta)
            img_new[i, j][2] = check(alpha*img_new[i, j][2] + beta)

    return img_new

img = cv2.imread('./Lenna.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

start_time = time.time()

#alpha < 1 thì độ tương phản giảm, ngược lại độ tương phản tăng
#có thể cho alpha = 0.5 và beta = 10 để làm ảnh tối hơn
img_br_c = brightness_contrast(img, 0.5, 10)
img_br = change_brightness(img, 50)
img_c = change_contrast(img, 2)

#hiển thị ảnh
plt.subplot(3,3,1), plt.imshow(img_br_c)
plt.title('Bright_Contrast'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,2), plt.imshow(img_br)
plt.title('Brightness'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,3), plt.imshow(img_c, cmap = 'gray')
plt.title('Contrast'), plt.xticks([]), plt.yticks([])

end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

plt.show()