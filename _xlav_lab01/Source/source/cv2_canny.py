import cv2, time
import matplotlib.pyplot as plt 

img = cv2.imread('./Lenna.jpg', 0)


start_time = time.time()

#gọi các hàm có sẵn để thực hiện
img_me = cv2.medianBlur(img, 5)
img_gauss = cv2.GaussianBlur(img, (5,5), 1)
img_gauss = cv2.Canny(img_gauss, 100, 200)
img_me = cv2.Canny(img_me, 100, 200)

plt.subplot(1,3,1), plt.imshow(img_gauss, cmap='gray')
plt.title("Canny_threshold_GaussianBlur_edge")
plt.subplot(1,3,3), plt.imshow(img_me, cmap='gray')
plt.title("Canny_threshold_MedianBlur_edge")

end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

plt.show()