#from scipy import ndimage để có thể gọi hàm tính tích chập
import numpy as np
import math

#hàm để tính tích chập
def convolve(img, kernel):

    H = (kernel.shape[0] - 1) // 2
    W = (kernel.shape[1] - 1) // 2

    #loại bỏ những giá trị 0 ở viền của ảnh kết quả
    #out = np.zeros((img.shape[0] - kernel.shape[0] + 1, img.shape[1] - kernel.shape[1] + 1))
    #hoặc để padding zero ở viền ảnh gốc để đảm bảo ảnh đầu ra không bị thu nhỏ
    out = np.zeros((img.shape[0], img.shape[1]))

    #Hai vòng lặp ngoài cùng biến đếm i cho hàng, j cho cột, thay đổi để dịch chuyển ma trận mặt nạ kernel
    for i in range(H, img.shape[0] - H):
        for j in range(W, img.shape[1] - W):
            sum = 0
            #Hai vòng lặp k, l thực hiện phép dot product giữa ma trận cửa sổ với kernel
            for k in range(-H, H + 1):
                for l in range(-W, W + 1):
                    a = img[i - k, j - l]
                    w = kernel[H + k, W + l]
                    sum += (w * a)              #g(x, y) =  f(x - i, y - j)*h(i, j) 
            out[i, j] = sum
    return out
    
#lập các mặt nạ để tích chập
def Sobel_filter(img, direction):
    if(direction == 'y'):
        Gy = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
        Gy = (1/4) * Gy
        Res = convolve(img, Gy) #tính tích chập
        #Res = ndimage.convolve(img, Gy) #nếu dùng thư viện
        #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if(direction == 'x'):
        Gx = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
        Gx = (1/4) * Gx
        Res = convolve(img, Gx)
        #Res = ndimage.convolve(img, Gx)
        #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

    return Res

#thực hiện giống như trên
def Roberts_filter(img, direction):
    if(direction == 'y'):
        Gy = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        
        Res = convolve(img, Gy) #tính tích chập
        #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if(direction == 'x'):
        Gx = np.array([[0 , 0, 0], [0, 1, 0], [-1, 0, 0]])
        
        Res = convolve(img, Gx)
        #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

    return Res

#thực hiện giống như trên
def Prewitt_filter(img, direction):
    if(direction == 'y'):
        Gy = np.array([[-1 , 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Gy = (1/3) * Gy
        Res = convolve(img, Gy) #tính tích chập
        #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if(direction == 'x'):
        Gx = np.array([[1 , 1, 1], [0, 0, 0], [-1, -1, -1]])
        Gx = (1/3) * Gx
        Res = convolve(img, Gx)
        #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

    return Res

#thực hiện giống như trên
def FreiChen_filter(img, direction):
    if(direction == 'y'):
        Gy = np.array([[-1 , 0, 1], [-math.sqrt(2), 0, math.sqrt(2)], [-1, 0, 1]])
        Gy = (1/(2 + math.sqrt(2))) * Gy
        Res = convolve(img, Gy) #tính tích chập
        #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if(direction == 'x'):
        Gx = np.array([[1 , math.sqrt(2), 1], [0, 0, 0 ], [-1, -math.sqrt(2), -1]])
        Gx = (1/(2 + math.sqrt(2))) * Gx
        Res = convolve(img, Gx)
        #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

    return Res

#thực hiện giống như trên
def Laplace_filter(img, direction):
    if(direction == 'negative'):
        neg = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        
        Res = convolve(img, neg) #tính tích chập
        #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if(direction == 'normal'):
        nor = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        
        Res = convolve(img, nor)
        #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

    return Res