from blur import *
import cv2, time

img = cv2.imread('./Lenna.jpg', cv2.IMREAD_COLOR)
img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#hàm tính thời gian thực hiện
start_time = time.time()
img_cvt = Median_blur(img_cvt)

plt.imshow(img_cvt)
plt.title("Median Filter")
end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

plt.show()
