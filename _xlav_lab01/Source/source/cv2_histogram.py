import cv2, time
import matplotlib.pyplot as plt 

img = cv2.imread('./Hill.jpg', 0)

start_time = time.time()

img = cv2.equalizeHist(img)
plt.imshow(img, cmap='gray')
plt.title("Histogram Equalization")

end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

plt.show()