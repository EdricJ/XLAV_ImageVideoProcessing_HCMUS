import cv2, time
import matplotlib.pyplot as plt 

img = cv2.imread('./Lenna.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

start_time = time.time()

img1 = cv2.Sobel(img, cv2.CV_64F, 1, 0)
img2 = cv2.Sobel(img, cv2.CV_64F, 0, 1)
cv2.imwrite('outx_sobel_cv2.jpg', img1)
cv2.imwrite('outy_sobel_cv2.jpg', img2)

end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
