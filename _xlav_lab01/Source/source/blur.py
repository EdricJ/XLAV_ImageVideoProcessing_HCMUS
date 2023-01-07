import numpy as np 
import matplotlib.pyplot as plt 
import math, sys
from edge_detection import convolve

#xay dựng mặt nạ cho toán tử trung bình
h = np.ones((3,3))
h = (1/9)*h

def Filter(img):
    return convolve(img, h)

def Gauss(img, sigma):
    out = np.zeros(img.shape)
    sig = 1 / (math.sqrt(2*math.pi)*sigma)  #tính theo công thức ở bước đầu

    H = (h.shape[0] - 1) // 2
    W = (h.shape[1] - 1) // 2

    for i in range(H, img.shape[0] - H):
        for j in range(W, img.shape[1] - W):
            sum = 0
            #Hai vòng lặp k, l thực hiện phép dot product giữa ma trận cửa sổ với kernel
            for k in range(-H, H + 1):
                for l in range(-W, W + 1):
                    a = img[i - k, j - l]
                    w = h[k, l] = sig * math.expm1(-(math.pow(k, 2) + math.pow(l, 2))/(2*math.pow(sigma, 2))) #tính theo công thức ở bước 2
                    sum += (w * a)              #g(x, y) =  f(x - i, y - j)*h(i, j) 
            out[i, j] = sum
    return out

#làm trơn ảnh trên 3 kênh màu
def Gaussian_blur(cv_img, deviation = 0.4):
    H, W, Colr = cv_img.shape
    
    m_bufGss = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    #cường độ làm mờ
    m_dGaussian = deviation
    
    nHalf = int(max((m_dGaussian * 6 - 1) / 2, 1))
    nWstep = W * Colr
    
    pIn = cv_img.flatten()  #làm phẳng lại mảng chứa thông số của hình ảnh
    pTmp = (np.zeros(shape=(W,H,Colr))).flatten()
    pOut = (np.zeros(shape=(W,H,Colr))).flatten()
    
    n = 0
    while n <= nHalf:
      m_bufGss[nHalf - n] = m_bufGss[nHalf + n] = math.exp((-1 * n) * n / (2 * m_dGaussian*m_dGaussian)) # tính toán biến đổi để áp dụng cho từng pixel trong ảnh
      n += 1 

    for r in range(0, H):
      for c in range(0, W):
        for l in range(0, Colr):
          dSum = dGss = 0
    
          for n in range(-1 * nHalf, nHalf+1):
            px = c + n
            
            if 0 <= px < W:
                dSum += (pIn[nWstep * r + Colr * px + l] * m_bufGss[nHalf + n])
                dGss += m_bufGss[nHalf + n]
          
          pTmp[nWstep * r + Colr * c + l] = int(dSum / dGss)
    
    for r in range(0, H):
      for c in range(0, W):
        for l in range(0, Colr):
          dSum = dGss = 0
    
          for n in range(-1 * nHalf, nHalf+1):
            py = r + n
            if 0 <= py < H:
              absN = abs(n)
              dSum += (pTmp[nWstep * py + Colr * c + l] * m_bufGss[nHalf + absN])
              dGss += m_bufGss[nHalf + absN]
            
          pOut[nWstep * r + Colr * c + l] = int(dSum / dGss)
    np.set_printoptions(threshold=sys.maxsize)
    
    out = pOut.reshape(H,W,Colr)
    out = out.astype('uint8')
    
    return out

#hàm sắp xếp các phần tử
def Sort_array(a):
    flag = False
    for i in range (0, len(a), 1):
        for j in range (0, len(a) - i - 1, 1):
            if(a[j].any() > a[j + 1].any()):
                temp = a[j]
                a[j] = a[j + 1]
                a[j + 1] = temp
                flag = True
        
        if(flag == False):
          break

def Median_blur(img):
    final = img[:]
    '''
    for y in range(len(img)):
        for x in range(y):
            final[y,x]=img[y,x]
    '''
    #lập mặt nạ với các điểm lân cận
    members=[img[0,0]]*9
    for y in range(1,img.shape[0]-1):
        for x in range(1,img.shape[1]-1):
            members[0] = img[y-1,x-1]
            members[1] = img[y,x-1]
            members[2] = img[y+1,x-1]
            members[3] = img[y-1,x]
            members[4] = img[y,x]
            members[5] = img[y+1,x]
            members[6] = img[y-1,x+1]
            members[7] = img[y,x+1]
            members[8] = img[y+1,x+1]

            Sort_array(members)
            final[y,x]=members[4] #chọn điểm ở giữa sau khi đã sắp xếp tăng dần hoặc giảm dần

    return final




