import pickle

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# import scipy
# from scipy.optimize import nnls

# from skimage import color
# from skimage import morphology
# from skimage.transform import resize
# import skimage.io as io


import copy

# import cvxpy
# import cv2

# import seaborn as sns
from shapely.geometry import box, Polygon
from shapely.validation import make_valid

# def translate_line(line_coef, a_factor, c_factor):
#     # a_factor = (imgHE.shape[0] / imgHE128.shape[0]) * (imgHE128.shape[1] / imgHE.shape[1])
#     # c_factor = imgHE128.shape[1] / imgHE.shape[1]
    
#     return([line_coef[0] , -1, line_coef[2] * c_factor])

def translate_line(line_coef, a_factor, c_factor):
    # a_factor = (imgHE.shape[0] / imgHE128.shape[0]) / (imgHE128.shape[1] / imgHE.shape[1])
    # c_factor = imgHE128.shape[1] / imgHE.shape[1]
    
    return([line_coef[0] * a_factor, -1, line_coef[2] * c_factor])

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)  
    plt.show()

## slice config file

def generate_vertex(line1_coef, line2_coef, mask_shape):
    ## ax + by + c = 0
    # only in mask_shape[0] == mask_shape[1]
    if line1_coef[0] != 0:
        #y = 0
        point1 = [-1 * line1_coef[2] / line1_coef[0], 0]
        point3 = [-1 * (line1_coef[1]*mask_shape[1] + line1_coef[2]) / line1_coef[0], mask_shape[1]]
        point2 = [-1 * line2_coef[2] / line2_coef[0], 0]
        point4 = [-1 * (line2_coef[1]*mask_shape[1] + line2_coef[2]) / line2_coef[0], mask_shape[1]]
        
    else:
        #x = 0
        point1 = [0, -1 * line1_coef[2] / line1_coef[1]]
        point3 = [mask_shape[0], -1 * (line1_coef[0]*mask_shape[0] + line1_coef[2]) / line1_coef[1]]
        point2 = [0, -1 * line2_coef[2] / line2_coef[1]]
        point4 = [mask_shape[0], -1 * (line2_coef[0]*mask_shape[0] + line2_coef[2]) / line2_coef[1]]
    
    pol1_xy = [point1, point3, point4, point2]
    return(pol1_xy)

def cal_intersection(pol1_xy, pol2_xy):
    
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    
    # valid_shape = make_valid(invalid_shape)
    
    if polygon1_shape.intersection(polygon2_shape):
        polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    else:
        polygon_intersection = 0
    
    return(polygon_intersection)
    
def grid_poly_search(mask_shape, pol1_xy):
    mask_shape_x, mask_shape_y = mask_shape
    weight_line = np.zeros(mask_shape)
    
    for i in range(mask_shape_x):
        for j in range(mask_shape_y):
            pol2_xy = [[i-0.5, j-0.5], [i-0.5, j+0.5], [i+0.5, j+0.5], [i+0.5, j-0.5]]
            weight = cal_intersection(pol1_xy, pol2_xy)
            weight_line[i, j] = weight
    
    return(weight_line.flatten())

def traverse_generate_vertex(line1_coef, line2_coef, max_xy):
    ## ax + by + c = 0
    # only in mask_shape[0] == mask_shape[1]
    tmp_shape = [max_xy, max_xy]
    if line1_coef[0] != 0:
        #y = 0
        point1 = [-1 * line1_coef[2] / line1_coef[0], 0]
        point3 = [-1 * (line1_coef[1]*tmp_shape[1] + line1_coef[2]) / line1_coef[0], tmp_shape[1]]
        point2 = [-1 * line2_coef[2] / line2_coef[0], 0]
        point4 = [-1 * (line2_coef[1]*tmp_shape[1] + line2_coef[2]) / line2_coef[0], tmp_shape[1]]
        
    else:
        #x = 0
        point1 = [0, -1 * line1_coef[2] / line1_coef[1]]
        point3 = [tmp_shape[0], -1 * (line1_coef[0]*tmp_shape[0] + line1_coef[2]) / line1_coef[1]]
        point2 = [0, -1 * line2_coef[2] / line2_coef[1]]
        point4 = [tmp_shape[0], -1 * (line2_coef[0]*tmp_shape[0] + line2_coef[2]) / line2_coef[1]]
    
    pol1_xy = [point1, point3, point4, point2]
    return(pol1_xy)

def traverse_poly_search_radis(point_x, point_y, pol1_xy, x_radis=0.5, y_radis=0.5):
    
    point_length = point_x.shape[0]
    weight_line = np.zeros(point_length)
    
    for m in range(point_length):
        i = point_x[m]
        j = point_y[m]
        pol2_xy = [[i-x_radis, j-y_radis], [i-x_radis, j+y_radis], [i+x_radis, j+y_radis], [i+x_radis, j-y_radis]]
        weight = cal_intersection(pol1_xy, pol2_xy)
        weight_line[m] = weight
    
    return(weight_line.flatten())

def transfer_mask(value, mask, adata):
    transfer = np.zeros(mask.shape)
    for index in range(adata.obs.shape[0]):
        transfer[adata.obs['y'][index], adata.obs['x'][index]] = value[index]
    
    return(transfer.flatten())

def transfer_mask_rw(value, mask, point_row, point_col):
    transfer = np.zeros(mask.shape)
    for index in range(len(point_row)):
        transfer[point_row[index], point_col[index]] = value[index]
    
    return(transfer.flatten())

### slice segment type based on design
def seg_count(design_i, segments_fz_i_flatten, class_num):
    design_i_bool = design_i > 0
    design_i_True = design_i[design_i_bool]

    segment_list = segments_fz_i_flatten[design_i_bool]
    segment_list_unique = np.unique(segment_list)

    segment_count = {}
    for i in range(class_num):
        segment_count[str(i)] = 0

    for seg_i in segment_list_unique:
        count = np.sum(design_i_True[segment_list == seg_i])
        segment_count[str(seg_i)] = count

    return (np.array(list(segment_count.values())))

def iterative_channel_count(segments_list, design_x, design_y, mask):
    channel_count_dat_list = []
    for segment_index_i in segments_list:
        segments_fz_i = segment_index_i.astype(int)#transfer_mask(np.array(adata.obs[segment_index_i]), mask, adata).astype(int)
        segments_fz_i_flatten = segments_fz_i.flatten()
        class_num = int(segments_fz_i_flatten.max() + 1)

        channel_count = []
        design_tmp = design_x
        for design_i in design_tmp:
            tmp = seg_count(design_i, segments_fz_i_flatten, class_num)
            channel_count.append(tmp)

        design_tmp = design_y
        for design_i in design_tmp:
            tmp = seg_count(design_i, segments_fz_i_flatten, class_num)
            channel_count.append(tmp)

        channel_count_dat = pd.DataFrame(channel_count)
        channel_count_dat_list.append(channel_count_dat)
    
    return(channel_count_dat_list)

def get_HE_distribution(y):
    
    yn1 = []
    yp1 = []
    
    for i in range(len(y)):
        
        if i % 2 == 0:
            if i == 0:
                yn1.append(np.nan)
                yp1.append(y[i] / y[i+1])
                
            elif i == len(y)-1:
                if y[i-1] == 0:
                    yn1.append(0)
                else:
                    yn1.append(y[i] / y[i-1])
                               
                yp1.append(np.nan)
                    
            else:
                min_y = min(y[i - 1], y[i + 1])
                max_y = max(y[i - 1], y[i + 1])
                
                if y[i] > min_y and y[i] < max_y:
                ## coef to 1
                    coef1 = (y[i] - min_y) / (max_y - min_y)
                    if y[i + 1] == max_y:
                        yn1.append(1 - coef1)
                        yp1.append(coef1)
                    else:
                        yn1.append(coef1)
                        yp1.append(1 - coef1)
                elif y[i] <= min_y:
                    if y[i] == 0:
                        yn1.append(0)
                        yp1.append(0)
                    elif y[i + 1] == max_y:
                        yn1.append(y[i] / y[i-1])
                        yp1.append(0)
                    else:
                        yn1.append(0)
                        yp1.append(y[i] / y[i+1])
                elif y[i] >= max_y:
                    if y[i + 1] == max_y:
                        yn1.append(0)
                        yp1.append(y[i] / y[i+1])
                    else:
                        yn1.append(y[i] / y[i-1])
                        yp1.append(0)
                    
                
        elif i % 2 == 1:
            yn1.append(np.nan)
            yp1.append(np.nan)
    
    return(np.array(yn1), np.array(yp1))
