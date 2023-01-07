import math
import cv2
import numpy as np

def Hough_line(img, rho=1, theta=np.pi/180, threshold=100, rho_delta=1, px_delta=5, min_line_length_percent=0.15):
    img_height, img_width = img.shape[:2]
    diagonal_length = int(math.sqrt(img_height*img_height + img_width*img_width))
    
    print('[My Hough] Img Height: %d | Img Width: %d | Img Diagonal Length: %d' % (img_height, img_width, diagonal_length))
    
    num_rho = int(diagonal_length / rho)
    num_theta = int(np.pi / theta)
    
    edge_matrix = np.zeros([2*num_rho+1, num_theta]) # dim: num_rho x num_theta
    
    print('[My Hough] Edge Matrix Dim: %d x %d' % (edge_matrix.shape[0], edge_matrix.shape[1]))
    
    idx	= np.squeeze(cv2.findNonZero(img)) # dim: 4468 x 2 (example, number of rows = number of white pixel on image processed by canny edge algorithm!)
    
    range_theta = np.arange(0, np.pi, theta)   
    theta_matrix = np.stack((np.cos(np.copy(range_theta)), np.sin(np.copy(range_theta))), axis=-1) # dim: 180 x 2
    
    vote_matrix = np.dot(idx, np.transpose(theta_matrix)) # => (4468 x 2) * (180 x 2)T = (4468 x 2) * (2 x 180) = 4468 x 180
    print('[My Hough] Vote Matrix Dim: %d x %d' % (vote_matrix.shape[0], vote_matrix.shape[1]))
    
    # loop on vote matrix and accumulate values on edge matrix
    for vr in range(vote_matrix.shape[0]):
        for vc in range(vote_matrix.shape[1]):
            rho_pos = int(round(vote_matrix[vr, vc]))+num_rho
            edge_matrix[rho_pos, vc] += 1
    
    print('[My Hough] Sum of Edge Matrix = %d | Max = %d | Min = %d' % (int(np.sum(edge_matrix)), int(np.max(edge_matrix)), int(np.min(edge_matrix))))
    
    line_idx = np.where(edge_matrix > threshold)
    
    rho_values = list(line_idx[0])
    rho_values = [r-num_rho for r in rho_values]
    theta_values = list(line_idx[1])
    theta_values = [t/180.0*np.pi for t in theta_values]
    
    line_idx = list(zip(rho_values, theta_values))
    # line_idx: [(-626, 2.6354471705114375), (-625, 2.6354471705114375), (-304, 2.548180707911721), (-11, 2.2165681500327987), (39, 0.0), (136, 1.5707963267948966), (319, 1.5707963267948966), (320, 1.5707963267948966), (341, 0.0), (408, 1.5707963267948966), (419, 1.5707963267948966), (422, 1.5533430342749532), (438, 0.0), (562, 1.5707963267948966), (588, 1.5707963267948966), (623, 0.0), (624, 0.9250245035569946), (733, 0.2792526803190927), (772, 0.41887902047863906), (773, 0.41887902047863906), (1017, 0.593411945678072)]
    
    #line_idx = [[li] for li in line_idx]
    
    line_px_dict = {}
    # line_px_dict: {
    #    (rho_0, theta_0): [(x0,y0), (x1,y1), ...],
    #    ...
    #}
    for (rho, theta) in line_idx:
        line_px_dict[(rho, theta)] = [] # init empty list for the key (rho, theta)
        for row_idx in range(idx.shape[0]): # idx dim: 4468 x 2
            point_x, point_y = idx[row_idx,0], idx[row_idx,1]
            rho_point = point_x*np.cos(theta) + point_y*np.sin(theta) # ρ=xcosθ+ysinθ
            if abs(int(rho_point) - rho) <= rho_delta:
                line_px_dict[(rho, theta)].append((point_x, point_y))
            pass
        pass
    
    # find (xmin, ymin) and (xmax, ymax) of every (rho, theta) in line_px_dict
    line_result = []
    for (rho, theta) in line_px_dict.keys():
        point_list = line_px_dict[(rho, theta)] # [(x0,y0), (x1,y1), ...]
        point_list.sort(key=lambda tup: tup[0]) # sort by x
        
        pf_idx = 0
        pfrom = point_list[pf_idx]
        pcurrent = pfrom
        for p_idx in range(len(point_list)):
            pto = point_list[p_idx]
            if abs(pto[0]-pcurrent[0]) <= px_delta and abs(pto[1]-pcurrent[1]) <= px_delta and p_idx != len(point_list)-1:
                pcurrent = pto
            else:
                pt_idx = p_idx-1 if p_idx != len(point_list)-1 else p_idx
                pto = point_list[pt_idx]
                line_length_percent = (pt_idx-pf_idx)/len(point_list)
                
                if line_length_percent >= min_line_length_percent:
                    line_result.append([pfrom, pto])
                else:
                    pass
                pf_idx = p_idx
                pfrom = point_list[p_idx]
                pcurrent = pfrom

    return line_result

def main():
    # read image
    img = cv2.imread('./Lenna.jpg')
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # color -> gray
    edges = cv2.Canny(gray, 50, 150, apertureSize=3) # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny
    cv2.imwrite('geo_canny.jpg', edges)
    
    # IMPLEMENT HOUGH ALGORITHM MYSELF!
    lines = Hough_line(edges, rho=1, theta=np.pi/180, threshold=100)
    for line in lines:
        (x1, y1) = line[0]
        (x2, y2) = line[1]
        cv2.line(img, (x1,y1), (x2,y2),(255,0,143), 3)    
    cv2.imwrite('geo_shape.jpg',img)

main()