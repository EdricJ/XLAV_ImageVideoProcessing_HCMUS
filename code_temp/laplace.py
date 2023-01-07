import sys
import cv2 as cv
from scipy import ndimage 
import numpy as np

#mặt nạ để tích chập
def Laplace(img, direction):
    if(direction == 'nega'):
        ne = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        Res = ndimage.convolve(img, ne) #thư viện để tính tích chập
        #Res = ndimage.convolve(img, ne, mode='constant', cval=0.0)
    if(direction == 'nor'):
        no = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        
        Res = ndimage.convolve(img, no)
        #Res = ndimage.convolve(img, no, mode='constant', cval=0.0)

    return Res

def main(argv):
    # [variables]
    # Declare the variables we are going to use
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]
    # [load]
    imageName = argv[0] if len(argv) > 0 else './Lenna.jpg'
    src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR) # Load an image
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default lena.jpg]')
        return -1
    # [load]
    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = cv.GaussianBlur(src, (3, 3), 0)
    # [reduce_noise]
    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # [convert_to_gray]
    # Create Window
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    # [laplacian]
    # Apply Laplace function
    dst = Laplace(src_gray, "ne")
    #dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [laplacian]
    # [convert]
    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)
    # [convert]
    # [display]
    cv.imshow(window_name, abs_dst)
    cv.waitKey(0)
    # [display]
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])