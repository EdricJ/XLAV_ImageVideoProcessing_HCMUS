import cv2, time
import matplotlib.pyplot as plt 

img = cv2.imread('./Lenna.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


start_time = time.time()

img = cv2.GaussianBlur(img, (5,5), 1)
plt.imshow(img)
plt.title("Gaussian Filter")

end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

plt.show()