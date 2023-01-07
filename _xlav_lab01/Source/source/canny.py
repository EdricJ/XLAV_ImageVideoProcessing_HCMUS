import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from edge_detection import Sobel_filter
from blur import *

#đọc vào ảnh
img = cv2.imread('./Lenna.jpg', cv2.IMREAD_COLOR)
#chuyển lại về theo thứ thự của các kênh màu
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Hàm để lọc ngưỡng
def Normalize(img):
    #img = (img - np.min(img) * 255) / (np.max(img) - np.min(img))
    return (img - np.min(img) * 255) / (np.max(img) - np.min(img))

#Hàm giúp xác định vị trí cạnh
#loại bỏ các pixel ở vị trí không phải cực đại toàn cục
def NonMaxSupWithInterpol(Gmag, Grad, Gx, Gy):
    #tạo mảng 0 với kích thước giống với kích thước với mảng chứa các điểm trong khoảng cách
    A = np.zeros(Gmag.shape, np.uint8)
    
    #The edge direction angle is rounded to one of four angles representing vertical, horizontal, and the two diagonals (0°, 45°, 90°, 135° and 180°)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= 0 and Grad[i,j] <= 45) or (Grad[i,j] < -135 and Grad[i,j] >= -180)):
                bot_Y = np.array([Gmag[i,j+1], Gmag[i+1,j+1]])
                top_Y = np.array([Gmag[i,j-1], Gmag[i-1,j-1]])
                x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                #nếu nằm trong vùng giữa 2 ngưỡng trên và ngưỡng dưới
                rule_top = (top_Y[1]-top_Y[0])*x_est+top_Y[0]
                rule_bot = (bot_Y[1]-bot_Y[0])*x_est+bot_Y[0]
                
                #Nếu là cực đại giữ pixel đó lại
                if (Gmag[i,j] >= rule_bot and Gmag[i,j] >= rule_top):
                      A[i,j] = Gmag[i,j]
                #không phải là cực đại lân cận, ta sẽ set độ lớn gradient của nó về zero
                else:
                    A[i,j] = 0

            if((Grad[i,j] > 45 and Grad[i,j] <= 90) or (Grad[i,j] < -90 and Grad[i,j] >= -135)):
                bot_Y = np.array([Gmag[i+1,j] ,Gmag[i+1,j+1]])
                top_Y = np.array([Gmag[i-1,j] ,Gmag[i-1,j-1]])
                x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                
                rule_top = (top_Y[1]-top_Y[0])*x_est+top_Y[0]
                rule_bot = (bot_Y[1]-bot_Y[0])*x_est+bot_Y[0]
                #Phân ngưỡng để loại kết quả dư thừa
                if (Gmag[i,j] >= rule_bot and Gmag[i,j] >= rule_top):
                    A[i,j] =Gmag[i,j]
                else:
                    A[i,j] = 0

            if((Grad[i,j] > 90 and Grad[i,j] <= 135) or (Grad[i,j] < -45 and Grad[i,j] >= -90)):
                bot_Y = np.array([Gmag[i+1,j] ,Gmag[i+1,j-1]])
                top_Y = np.array([Gmag[i-1,j] ,Gmag[i-1,j+1]])
                x_est = np.absolute(Gx[i,j]/Gmag[i,j])

                rule_top = (top_Y[1]-top_Y[0])*x_est+top_Y[0]
                rule_bot = (bot_Y[1]-bot_Y[0])*x_est+bot_Y[0]
                if (Gmag[i,j] >= rule_bot and Gmag[i,j] >= rule_top):
                    A[i,j] =Gmag[i,j]
                else:
                    A[i,j] = 0

            if((Grad[i,j] > 135 and Grad[i,j] <= 180) or (Grad[i,j] < 0 and Grad[i,j] >= -45)):
                bot_Y = np.array([Gmag[i,j-1] ,Gmag[i+1,j-1]])
                top_Y = np.array([Gmag[i,j+1] ,Gmag[i-1,j+1]])
                x_est = np.absolute(Gy[i,j]/Gmag[i,j])

                rule_top = (top_Y[1]-top_Y[0])*x_est+top_Y[0]
                rule_bot = (bot_Y[1]-bot_Y[0])*x_est+bot_Y[0]

                if (Gmag[i,j] >= rule_bot and Gmag[i,j] >= rule_top):
                    A[i,j] =Gmag[i,j]
                else:
                    A[i,j] = 0
    
    return A

#đệ quy qua mọi cạnh mạnh và tìm tất cả các cạnh yếu được kết nối
#threshold to get rid of the grey areas and get solid edges
def Thresh_Hyst(img):
    highRatio =0.32
    lowRatio = 0.30
    image = np.copy(img)
    h = int(image.shape[0])
    w = int(image.shape[1])
    high = np.max(image) * highRatio
    low = high * lowRatio    
    
    #thực hiện cho đến khi số lượng cạnh mạnh không thay đổi, tức là tất cả các cạnh yếu được kết nối với các cạnh mạnh đã được tìm thấy
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(image[i,j] > high):
                image[i,j] = 1
            elif(image[i,j] < low):
                image[i,j] = 0
            else:
                if((image[i-1,j-1] > high) or (image[i-1,j] > high) or
                    (image[i-1,j+1] > high) or (image[i,j-1] > high) or
                    (image[i,j+1] > high) or (image[i+1,j-1] > high) or
                    (image[i+1,j] > high) or (image[i+1,j+1] > high)):
                    image[i,j] = 1
    
    #Thao tác này được thực hiện để loại bỏ tất cả các cạnh yếu không được kết nối với các cạnh mạnh
    #image = (image == 1) * image
    
    return (image == 1) * image

def Show_image():
    #Làm mờ hình ảnh với gaussian filter, giúp phát hiện cạnh và loại bỏ nhiễu 
    #gaussian_filter(img, sigma=1) 
    img_guassian_filter = Gaussian_blur(img, 1)
    #convert color image to grayscale to help extraction of edges and plot it
    img_grey = np.dot(img_guassian_filter[...,:3], [0.299, 0.587, 0.114])

    #tính gradient và hướng gradient
    gx = Sobel_filter(img_grey, 'x')
    gy = Sobel_filter(img_grey, 'y')
    #print(gx.shape)
    #print(gy.shape)

    #trả về tập các điểm nằm trong khoảng cách giữa 2 ngưỡng để xác định chính xác cạnh 
    Mag = np.hypot(gx,gy)
    #print(Mag.shape)

    #tính toán hướng gradient (hướng của cạnh)
    Gradient = np.degrees(np.arctan2(gy,gx))
    #or
    #Gradient = np.arctan2(gy, gx) * 180 / np.pi
    #print(Gradient)

    NMS = NonMaxSupWithInterpol(Mag, Gradient, gx, gy)
    NMS = Normalize(NMS)

    Final_image = Thresh_Hyst(NMS)

    #hiển thị đều 3 ảnh ở 1 hàng
    plt.subplot(3,3,1), plt.imshow(img)
    plt.title('Color'), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,2), plt.imshow(img_guassian_filter)
    plt.title('Gaussianfilter'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(3,3,3), plt.imshow(img_grey, cmap = 'gray')
    plt.title('Convert2GrayScale'), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,4), plt.imshow(gx, cmap = 'gray')
    plt.title('SobelWithX'), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,5), plt.imshow(gy, cmap = 'gray')
    plt.title('SobelWithY'), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,6), plt.imshow(Mag, cmap = 'gray')
    plt.title('HighPot_distance'), plt.xticks([]), plt.yticks([])
    cv2.imwrite('output1_canny.jpg', Mag)

    plt.subplot(3,3,7), plt.imshow(NMS, cmap = 'gray')
    plt.title('Canny_edge'), plt.xticks([]), plt.yticks([])
    plt.imsave('output2_canny.jpg', NMS, cmap = 'gray')

    plt.subplot(3,3,8), plt.imshow(Final_image, cmap = 'gray')
    plt.title('Canny_threshold'), plt.xticks([]), plt.yticks([])
    plt.imsave('output3_canny.jpg', Final_image, cmap = 'gray')


start_time = time.time()
Show_image()
end_time = time.time()

elapsed_time = end_time - start_time
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

plt.show()


