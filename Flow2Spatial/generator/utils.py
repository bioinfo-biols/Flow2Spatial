import pickle

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import scipy
from scipy.optimize import nnls

# from skimage import color
# from skimage import morphology
from skimage.transform import resize
# import skimage.io as io


import copy

import cvxpy
# import cv2

# import seaborn as sns

# from anndata import AnnData
# import scanpy as sc
# import squidpy as sq

import os

def iter_IPF_a(reconstruct_data_2array, y1_row, y1_col, design, orition = 1):
    # orition = 1
    tmp = []
    for i in range(design.shape[0]):
        
        tmp.append(sum(reconstruct_data_2array*design[i]))

    if orition == 1:
        y_tmp = y1_row
    else:
        y_tmp = y1_col
        
    new_reconstruct = np.zeros(reconstruct_data_2array.shape)
    
    for i in range(len(tmp)):
        if np.sum(design[i]>0) < 3:
            new_reconstruct[design[i]>0] = reconstruct_data_2array[design[i]>0]
            continue
        elif tmp[i] > 0:
            new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0] / tmp[i]
        else:
            pass#new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0]
    
    return(new_reconstruct)
    
def iter_IPF(reconstruct_data_2array, y1_row, y1_col, design, orition = 1):
  
    # orition = 1
    tmp = []
    for i in range(design.shape[0]):
        
        tmp.append(sum(reconstruct_data_2array*design[i]))

    
    y_tmp = y1_row if orition == 1 else y1_col

    new_reconstruct = np.zeros(reconstruct_data_2array.shape)
    
    for i in range(len(tmp)):

        if tmp[i] > 0:
            if orition == 1:
                new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0] / tmp[i]
            else:
                new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0] / tmp[i]
        else:
            if orition == 1:
                new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0]
            else:
                new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0]

    return(new_reconstruct)

def iter_IPF1(reconstruct_data_2array, y1_row, y1_col, design, orition = 1):
  
    # orition = 1
    tmp = []
    for i in range(design.shape[0]):
        
        tmp.append(sum(reconstruct_data_2array*design[i]))

    
    y_tmp = y1_row if orition == 1 else y1_col

    new_reconstruct = np.zeros(reconstruct_data_2array.shape)
    
    def iter_recon3(y_tmp_now, tmp_now, m, n, t, orition, new_reconstruct=new_reconstruct, reconstruct_data_2array=reconstruct_data_2array):
        if tmp_now > 0:
            if orition == 1:
                new_reconstruct[(design[m]+design[n]+design[t])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n]+design[t])>0] / tmp_now
            else:
                new_reconstruct[(design[m]+design[n]+design[t])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n]+design[t])>0] / tmp_now
        else:
            if orition == 1:
                new_reconstruct[(design[m]+design[n]+design[t])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n]+design[t])>0]
            else:
                new_reconstruct[(design[m]+design[n]+design[t])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n]+design[t])>0]
        return(new_reconstruct)
    
    for i in range(len(tmp)):
        if (orition == 1) and (i == len(tmp) - 3):
            tmp_now = tmp[i] + tmp[i+1] + tmp[i+2]
            y_tmp_now = y_tmp[i] + y_tmp[i+1] + tmp[i+2]
            new_reconstruct = iter_recon3(y_tmp_now, tmp_now, i, i+1, i+2, orition)
            i = i + 2

        else:   
            if tmp[i] > 0:
                if orition == 1:
                    new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0] / tmp[i]
                else:
                    new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0] / tmp[i]
            else:
                if orition == 1:
                    new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0]
                else:
                    new_reconstruct[design[i]>0] = y_tmp[i] * reconstruct_data_2array[design[i]>0]
    
    return(new_reconstruct)

def iter_IPF2(reconstruct_data_2array, y1_row, y1_col, design, orition = 1):
  
    # orition = 1
    tmp = []
    for i in range(design.shape[0]):
        tmp.append(sum(reconstruct_data_2array*design[i]))

    y_tmp = y1_row if orition == 1 else y1_col

    new_reconstruct = np.zeros(reconstruct_data_2array.shape)#
    
    len_tmp = len(tmp)
    iter_list = list(reversed([list(range(len_tmp))[(len_tmp + (~i, i)[i%2]) // 2] for i in range(len_tmp)]))
    
    def iter_recon3(y_tmp_now, tmp_now, m, n, t, orition, new_reconstruct=new_reconstruct, reconstruct_data_2array=reconstruct_data_2array):
        if tmp_now > 0:
            if orition == 1:
                new_reconstruct[(design[m]+design[n]+design[t])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n]+design[t])>0] / tmp_now
            else:
                new_reconstruct[(design[m]+design[n]+design[t])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n]+design[t])>0] / tmp_now
        else:
            if orition == 1:
                new_reconstruct[(design[m]+design[n]+design[t])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n]+design[t])>0]
            else:
                new_reconstruct[(design[m]+design[n]+design[t])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n]+design[t])>0]
        return(new_reconstruct)
    
    def iter_recon(y_tmp_now, tmp_now, m, n, orition, design=design, new_reconstruct=new_reconstruct, reconstruct_data_2array=reconstruct_data_2array):
        if tmp_now > 0:
            if orition == 1:
                new_reconstruct[(design[m]+design[n])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n])>0] / tmp_now
            else:
                new_reconstruct[(design[m]+design[n])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n])>0] / tmp_now
        else:
            if orition == 1:
                new_reconstruct[(design[m]+design[n])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n])>0]
            else:
                new_reconstruct[(design[m]+design[n])>0] = y_tmp_now * reconstruct_data_2array[(design[m]+design[n])>0]
        return(new_reconstruct)
    
    def iter_recon_s(y_tmp_now, tmp_now, m, orition, design=design, new_reconstruct=new_reconstruct, reconstruct_data_2array=reconstruct_data_2array):
        if tmp_now > 0:
            if orition == 1:
                new_reconstruct[design[m]>0] = y_tmp_now * reconstruct_data_2array[design[m]>0] / tmp_now
            else:
                new_reconstruct[design[m]>0] = y_tmp_now * reconstruct_data_2array[design[m]>0] / tmp_now
        else:
            if orition == 1:
                new_reconstruct[design[m]>0] = y_tmp_now * reconstruct_data_2array[design[m]>0]
            else:
                new_reconstruct[design[m]>0] = y_tmp_now * reconstruct_data_2array[design[m]>0]
        
        return(new_reconstruct)
    
    import copy
    iter_list1 = copy.copy(iter_list)
    
    for i in iter_list:
            
        if i < len_tmp / 2:
            if (i in iter_list1) and ((i + 1) in iter_list1):
                tmp_now = tmp[i] + tmp[i+1]
                y_tmp_now = y_tmp[i] + y_tmp[i+1]
                new_reconstruct = iter_recon(y_tmp_now, tmp_now, i, i+1, orition)
                
                iter_list1.remove(i)
                iter_list1.remove(i+1) 
                
            else:
                if i in iter_list1:
                    new_reconstruct = iter_recon_s(y_tmp[i], tmp[i], i, orition)
                    iter_list1.remove(i)
            
        else:
            if (i in iter_list1) and ((i - 1) in iter_list1):
                tmp_now = tmp[i] + tmp[i-1]
                y_tmp_now = y_tmp[i] + y_tmp[i-1]
                new_reconstruct = iter_recon(y_tmp_now, tmp_now, i, i-1, orition)
                
                iter_list1.remove(i)
                iter_list1.remove(i-1)
            else:
                if i in iter_list1:
                    new_reconstruct = iter_recon_s(y_tmp[i], tmp[i], i, orition)
                    iter_list1.remove(i)
    
    return(new_reconstruct)

# def only_run(y0,  y90, angle0_Design=angle0_Design, angle90_Design=angle90_Design, ref=ref, mask=mask, FDesign0=np.array(FDesign0), FDesign90=np.array(FDesign90)):
    
#     y = np.concatenate((y0,  y90))#np.array(y0 + y90)
#     ## first block solution
#     A = np.concatenate((angle0_Design, angle90_Design))
#     intensity, rnorm = nnls(A/1e4, y)
    
#     class_intensity = {}
#     for i in range(len(intensity)):
#         class_intensity[str(i)] = intensity[i]
        
#     num_data = ref.shape#95
#     reconstruct_data_2array = np.zeros(num_data)
#     for i in range(num_data[0]):
#         for j in range(num_data[1]):
#             reconstruct_data_2array[i][j] = class_intensity[str(ref[i][j])]
    
#     # im = plt.imshow(reconstruct_data_2array[40:82, 42:90])
#     # cbar = plt.colorbar(im)
#     # print(scipy.stats.spearmanr(maski[mask].flatten(), reconstruct_data_2array[mask].flatten()))
    
#     ## second convex optim
#     tmp_flatten = reconstruct_data_2array.flatten()#resize(, (95, 95))

#     x = cvxpy.Variable(tmp_flatten.shape[0])
#     A0 = FDesign0
#     A90 = FDesign90
#     back_x = x[~mask.flatten()]#x[tmp_flatten==0]#~mask.flatten()
    
#     tmp = 1
#     objective = cvxpy.Minimize(cvxpy.sum_squares(A0 @ x - y0) + cvxpy.sum_squares(A90 @ x - y90))# + tmp * cvxpy.sum_squares(x - tmp_flatten))#cvxpy.norm(x - tmp_flatten,1))#
#     constraints = [x >= 0., back_x == 0]
#     problem = cvxpy.Problem(objective, constraints)
#     problem.solve(warm_start=True)
    
#     tmp_array = x.value.reshape(ref.shape)
    
#     # tmp_array = tmp_array * tmp_array.sum() / (y0.sum() + y90.sum())
        
#     # im = plt.imshow(tmp_array, vmax = tmp_array.max() * 0.9)
#     # cbar = plt.colorbar(im)
#     # print(scipy.stats.spearmanr(maski[mask].flatten(), tmp_array[mask].flatten()))
    

#     reconstruct_data_IPF = tmp_array
    
#     return(tmp_flatten.reshape(ref.shape), reconstruct_data_IPF.reshape(ref.shape))
    
    
# def pipeline_run(y0,  y90, angle0_Design=angle0_Design, angle90_Design=angle90_Design, ref=ref, mask=mask, FDesign0=np.array(FDesign0), FDesign90=np.array(FDesign90)):
    
#     y = np.concatenate((y0,  y90))#np.array(y0 + y90)
#     ## first block solution
#     A = np.concatenate((angle0_Design, angle90_Design))
#     intensity, rnorm = nnls(A/1e4, y)
    
#     class_intensity = {}
#     for i in range(len(intensity)):
#         class_intensity[str(i)] = intensity[i]
        
#     num_data = ref.shape#95
#     reconstruct_data_2array = np.zeros(num_data)
#     for i in range(num_data[0]):
#         for j in range(num_data[1]):
#             reconstruct_data_2array[i][j] = class_intensity[str(ref[i][j])]
    
#     # im = plt.imshow(reconstruct_data_2array[40:82, 42:90])
#     # cbar = plt.colorbar(im)
#     # print(scipy.stats.spearmanr(maski[mask].flatten(), reconstruct_data_2array[mask].flatten()))
    
#     ## second convex optim
#     tmp_flatten = reconstruct_data_2array.flatten()#resize(, (95, 95))

#     x = cvxpy.Variable(tmp_flatten.shape[0])
#     A0 = FDesign0
#     A90 = FDesign90
#     back_x = x[~mask.flatten()]#x[tmp_flatten==0]#~mask.flatten()
    
#     tmp = 1
#     objective = cvxpy.Minimize(cvxpy.sum_squares(A0 @ x - y0) + cvxpy.sum_squares(A90 @ x - y90) + tmp * cvxpy.sum_squares(x - tmp_flatten))#cvxpy.norm(x - tmp_flatten,1))#
#     constraints = [x >= 0., back_x == 0]
#     problem = cvxpy.Problem(objective, constraints)
#     problem.solve(warm_start=True)
    
#     tmp_array = x.value.reshape(ref.shape)
    
#     tmp_array = tmp_array * tmp_array.sum() / (y0.sum() + y90.sum())
        
#     # im = plt.imshow(tmp_array, vmax = tmp_array.max() * 0.9)
#     # cbar = plt.colorbar(im)
#     # print(scipy.stats.spearmanr(maski[mask].flatten(), tmp_array[mask].flatten()))
    

#     reconstruct_data_IPF = tmp_array
    
#     ## IPF
#     y1_row = y0#insert_value(y[:96])
#     y1_col = y90#insert_value(y[96:])
    
#     reconstruct_data_IPF = copy.copy(tmp_array.flatten())
#     for i in range(10):
#         new_reconstruct_row = iter_IPF_a(reconstruct_data_IPF, y1_row, y1_col, FDesign0, orition = 1)
#         new_reconstruct = iter_IPF_a(new_reconstruct_row, y1_row, y1_col, FDesign90, orition = 0)

#         reconstruct_data_IPF = new_reconstruct
#         # print(scipy.stats.spearmanr(reconstruct_data_IPF[mask95].flatten(), msi_intensity[mask95].flatten()))
#         # plt.imshow(reconstruct_data_IPF)
    
#     # out_f = scipy.stats.spearmanr(reconstruct_data_IPF[mask95].flatten(), msi_intensity[mask95].flatten()).correlation
    
#     return(reconstruct_data_IPF)#.reshape(ref.shape))
    
def pipeline_run(y0,  y90, angle0_Design, angle90_Design, ref, mask, FDesign0, FDesign90):
    
    y = np.concatenate((y0,  y90))#np.array(y0 + y90)
    ## first block solution
    A = np.concatenate((angle0_Design, angle90_Design))
    intensity, rnorm = nnls(A, y)
    
    class_intensity = {}
    for i in range(len(intensity)):
        class_intensity[str(i)] = intensity[i]
        
    num_data = ref.shape#95
    reconstruct_data_2array = np.zeros(num_data)
    for i in range(num_data[0]):
        for j in range(num_data[1]):
            reconstruct_data_2array[i][j] = class_intensity[str(ref[i][j])]
    
    # im = plt.imshow(reconstruct_data_2array[40:82, 42:90])
    # cbar = plt.colorbar(im)
    # print(scipy.stats.spearmanr(maski[mask].flatten(), reconstruct_data_2array[mask].flatten()))
    
    ## second convex optim
    tmp_flatten = reconstruct_data_2array.flatten()#resize(, (95, 95))

    x = cvxpy.Variable(tmp_flatten.shape[0])
    A0 = FDesign0
    A90 = FDesign90
    back_x = x[~mask.flatten()]#x[tmp_flatten==0]#~mask.flatten()
    
    tmp = 1
    objective = cvxpy.Minimize(cvxpy.sum_squares(A0 @ x - y0) + cvxpy.sum_squares(A90 @ x - y90) + tmp * cvxpy.sum_squares(x - tmp_flatten))#cvxpy.norm(x - tmp_flatten,1))#
    constraints = [x >= 0., back_x == 0]
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(warm_start=True)
    
    tmp_array = x.value.reshape(ref.shape)
    tmp_array[tmp_array<0] = 0
    
    tmp_array = tmp_array * tmp_array.sum() / (y0.sum() + y90.sum())
        
    # im = plt.imshow(tmp_array, vmax = tmp_array.max() * 0.9)
    # cbar = plt.colorbar(im)
    # print(scipy.stats.spearmanr(maski[mask].flatten(), tmp_array[mask].flatten()))
    

    reconstruct_data_IPF = tmp_array
    
    ## IPF
    y1_row = y0#insert_value(y[:96])
    y1_col = y90#insert_value(y[96:])
    
    reconstruct_data_IPF = copy.copy(tmp_array.flatten())
    for i in range(10):
        new_reconstruct_row = iter_IPF_a(reconstruct_data_IPF, y1_row, y1_col, FDesign0, orition = 1)
        new_reconstruct = iter_IPF_a(new_reconstruct_row, y1_row, y1_col, FDesign90, orition = 0)

        reconstruct_data_IPF = new_reconstruct
        # print(scipy.stats.spearmanr(reconstruct_data_IPF[mask95].flatten(), msi_intensity[mask95].flatten()))
        # plt.imshow(reconstruct_data_IPF)
    
    # out_f = scipy.stats.spearmanr(reconstruct_data_IPF[mask95].flatten(), msi_intensity[mask95].flatten()).correlation
    
    return(reconstruct_data_IPF)#.reshape(ref.shape))


def HE_inter(y, yn1, yp1, len_y):
    inter_y = np.zeros((len_y))
    
    for i in range(len_y):
        if i % 2 == 0:
            if i == 0:
                inter_y[i] = yp1[i] * y[int(i/2)]
            elif i == len_y-1:
                inter_y[i] = yn1[i] * y[int(i/2) - 1]
            else:
                inter_y[i] = yn1[i] * y[int(i/2) - 1] + yp1[i] * y[int(i/2)]
            
        if i % 2 == 1:
            inter_y[i] = y[int(i/2)]
    
    return(inter_y)

def cal_length(instense_before, instense_after, length_i, length_before, length_after):
    min_len = min(length_before, length_after)
    max_len = max(length_before, length_after)
    if length_i > min_len and length_i < max_len:
        coef1 = (length_i - min_len) / (max_len - min_len)
        if length_after == max_len:
            out_y = coef1 * instense_after + (1 - coef1) * instense_before
        else:
            out_y = coef1 * instense_before + (1 - coef1) * instense_after
    elif length_i <= min_len:
        if min_len == 0:
            return(0)
        if length_before == min_len:
            out_y = instense_before * length_i / length_before
        else:
            out_y = instense_after * length_i / length_after
    elif length_i >= max_len:
        if length_before == max_len:
            out_y = instense_before * length_i / length_before
        else:
            out_y = instense_after * length_i / length_after
    
    return(out_y)         
    
    
def length_inter(y, length):
    
    inter_y = np.zeros(len(length))
    
    for i in range(len(inter_y)):
        if i % 2 == 0:
            inter_y[i] = y[int(i/2)]
        else:
            inter_y[i] = cal_length(y[int(i/2)], y[int(i/2)+1], length[i], length[i-1], length[i+1])

    return(inter_y)


def insert_value(y):
    n = 2 * len(y) - 1
    y_full = np.zeros((n))
    
    for i in range(n):
        if i % 2 == 0:
            y_full[i] = y[int(i/2)]
        else:
            y_full[i] = (y[int(i/2)] + y[int(i/2)+1] )/2.0
    
    return(y_full)

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

def normalize_outliners(dataset, descore=1):
    tmp_sum = dataset.sum(0)
    tmp_sum0 = tmp_sum[tmp_sum>0]
    tmp_sum0_mean = tmp_sum0.mean()
    tmp_sum0_zscore = scipy.stats.zscore(tmp_sum0)
    sum_num = np.sum(tmp_sum0_zscore > descore)

    if sum_num>0:
        index = tmp_sum.argsort()[-1 * sum_num:]
        dataset[:,index] = tmp_sum0_mean * dataset[:,index] / tmp_sum[index]

        return(dataset)

    else:
        return(dataset)
        

def similar_generator(seg_j, segments_list, channel_count_dat_list, design_row, design_col, channel_intensity_list_dat, img_mask_bool, length_coef, output_dir):
    # , segment_index=segment_index, channel_count_dat_list=channel_count_dat_list, design_row=design_row, design_col=design_col, channel_intensity_list_dat=channel_intensity_list_dat, img_mask_bool=img_mask_bool
    seg_j = int(seg_j)
    angle0_Design  = channel_count_dat_list[seg_j].iloc[0:len(design_row), :]
    angle90_Design = channel_count_dat_list[seg_j].iloc[len(design_row):(len(design_row) + len(design_col)), :]

    ref = segments_list[seg_j].astype(int)#transfer_mask(adata.obs[segment_index[seg_j]], img_mask_bool, adata).astype(int).reshape(img_mask_bool.shape)
    y0, y90 = length_coef

    out_list = []
    for i in range(channel_intensity_list_dat.shape[0]):
        # i = 0
        if i % 500 == 0:
            print(str(i))
            
        channel_intensity = channel_intensity_list_dat.iloc[i,:]
        Inten_x = channel_intensity[2:(2+int(len(design_row)/2) + 1)]#[::-1]#
        Inten_y = channel_intensity[(2+int(len(design_row)/2) + 1):]#[::-1]
        Inten_x_insert = length_inter(Inten_x.to_list(), y0)#length_inter(Inten_x.to_list(), y0)#insert_value(Inten_x.to_list())
        Inten_y_insert = length_inter(Inten_y.to_list(), y90)#length_inter(Inten_y.to_list(), y90)

        out = pipeline_run(Inten_x_insert,  Inten_y_insert, angle0_Design=angle0_Design, angle90_Design=angle90_Design, ref=ref, mask=img_mask_bool, FDesign0=np.array(design_row), FDesign90=np.array(design_col))
        out[out<0] = 0
        
        out_list.append(out)
    
    out_list_array = np.array(out_list)
    out_list_array1 = normalize_outliners(normalize_outliners(out_list_array), descore=3)
    
    with open(output_dir + '/histology' + str(seg_j), 'wb') as handle:#
        pickle.dump(out_list_array1, handle)
    
    # print(seg_j)
    return(seg_j)
