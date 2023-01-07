import numpy as np
import cv2, time
import matplotlib.pyplot as plt

img = cv2.imread('./Hill.jpg', 0)

#ham tinh histogram cua mot anh
def compute_histogram(img):
    hist = np.zeros((256,), np.uint8)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1    #tinh luoc do xam cua anh
    return hist

#ham can bang histogram
def equal_histogram(hist):
    equal = np.zeros_like(hist, np.float64)
    for i in range(len(equal)):
        equal[i] = hist[:i].sum()

    #print(equal)
    out = np.round((equal - equal.min())/(equal.max() - equal.min()) * 255)     #chuẩn hóa về đoạn [0,nG - 1]
    out = np.uint8(out)
    return out

def Histogram(img):
    hist = compute_histogram(img).ravel()
    out = equal_histogram(hist)

    h, w = img.shape[:2]

    for i in range(h):
        for j in range(w):
            img[i,j] = out[img[i,j]]       #lấy lại những điểm ảnh đã được cân bằng

    return img

start_time = time.time()

img_grey = Histogram(img)
plt.imshow(img_grey, cmap='gray')
plt.title("Histogram Equalization")

end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

plt.show()