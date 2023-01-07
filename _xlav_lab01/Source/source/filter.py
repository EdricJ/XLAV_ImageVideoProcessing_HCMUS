from blur import *
import cv2, time

img = cv2.imread('./Lenna.jpg', cv2.IMREAD_COLOR).astype('float64')

start_time = time.time()
img_cvt = np.dot(img[...,:3], [0.299, 0.587, 0.114])

img_cvt = Filter(img_cvt)

plt.imshow(img_cvt, cmap = 'gray')
plt.title("2D Filter")
end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

plt.show()
